use super::Operation;
use crate::{
    context::Context,
    ir::{
        attribute::DenseI32ArrayAttribute, Attribute, AttributeLike, Block, Identifier, Location,
        Region, Type, Value,
    },
    string_ref::StringRef,
    Error,
};
use mlir_sys::{
    mlirNamedAttributeGet, mlirOperationCreate, mlirOperationStateAddAttributes,
    mlirOperationStateAddOperands, mlirOperationStateAddOwnedRegions, mlirOperationStateAddResults,
    mlirOperationStateAddSuccessors, mlirOperationStateEnableResultTypeInference,
    mlirOperationStateGet, MlirOperationState,
};
use std::{
    marker::PhantomData,
    mem::{forget, transmute, ManuallyDrop},
};

/// An operation builder.
pub struct OperationBuilder<'c> {
    raw: MlirOperationState,
    _context: PhantomData<&'c Context>,
}

impl<'c> OperationBuilder<'c> {
    /// Creates an operation builder.
    pub fn new(name: &str, location: Location<'c>) -> Self {
        Self {
            raw: unsafe { mlirOperationStateGet(StringRef::new(name).to_raw(), location.to_raw()) },
            _context: Default::default(),
        }
    }

    /// Adds results.
    pub fn add_results(mut self, results: &[Type<'c>]) -> Self {
        unsafe {
            mlirOperationStateAddResults(
                &mut self.raw,
                results.len() as isize,
                results.as_ptr() as *const _,
            )
        }

        self
    }

    /// Adds operands.
    pub fn add_operands(mut self, operands: &[Value<'c, '_>]) -> Self {
        unsafe {
            mlirOperationStateAddOperands(
                &mut self.raw,
                operands.len() as isize,
                operands.as_ptr() as *const _,
            )
        }

        self
    }

    /// Adds operands with segment sizes for operations with variadic operands.
    ///
    /// Some MLIR operations have variadic or optional operands and require an
    /// `operandSegmentSizes` attribute to indicate how operands are grouped.
    /// This method takes operand segments and automatically:
    /// 1. Adds all operands in a flat list
    /// 2. Adds the `operandSegmentSizes` attribute with the segment sizes
    ///
    /// # Example
    ///
    /// For `gpu.launch` which has segments like:
    /// - asyncDependencies (variadic)
    /// - gridSizeX, gridSizeY, gridSizeZ (required)
    /// - blockSizeX, blockSizeY, blockSizeZ (required)
    /// - clusterSizeX, clusterSizeY, clusterSizeZ (optional)
    /// - dynamicSharedMemorySize (optional)
    ///
    /// ```ignore
    /// builder.add_operands_with_segment_sizes(
    ///     context,
    ///     &[
    ///         &[],                    // asyncDependencies (0)
    ///         &[grid_x],              // gridSizeX (1)
    ///         &[grid_y],              // gridSizeY (1)
    ///         &[grid_z],              // gridSizeZ (1)
    ///         &[block_x],             // blockSizeX (1)
    ///         &[block_y],             // blockSizeY (1)
    ///         &[block_z],             // blockSizeZ (1)
    ///         &[],                    // clusterSizeX (0)
    ///         &[],                    // clusterSizeY (0)
    ///         &[],                    // clusterSizeZ (0)
    ///         &[],                    // dynamicSharedMemorySize (0)
    ///     ],
    /// )
    /// ```
    pub fn add_operands_with_segment_sizes(
        mut self,
        context: &'c Context,
        segments: &[&[Value<'c, '_>]],
    ) -> Self {
        // Collect all operands into a flat list
        let all_operands: Vec<Value<'c, '_>> = segments.iter().flat_map(|s| s.iter().copied()).collect();

        // Add all operands
        if !all_operands.is_empty() {
            unsafe {
                mlirOperationStateAddOperands(
                    &mut self.raw,
                    all_operands.len() as isize,
                    all_operands.as_ptr() as *const _,
                )
            }
        }

        // Create segment sizes array
        let segment_sizes: Vec<i32> = segments.iter().map(|s| s.len() as i32).collect();
        let segment_attr = DenseI32ArrayAttribute::new(context, &segment_sizes);

        // Add operandSegmentSizes attribute
        unsafe {
            mlirOperationStateAddAttributes(
                &mut self.raw,
                1,
                &[mlirNamedAttributeGet(
                    Identifier::new(context, "operandSegmentSizes").to_raw(),
                    segment_attr.to_raw(),
                )] as *const _,
            )
        }

        self
    }

    /// Adds regions.
    pub fn add_regions<const N: usize>(mut self, regions: [Region<'c>; N]) -> Self {
        unsafe {
            mlirOperationStateAddOwnedRegions(
                &mut self.raw,
                regions.len() as isize,
                regions.as_ptr() as *const _,
            )
        }

        forget(regions);

        self
    }

    /// Adds regions in a [`Vec`](std::vec::Vec).
    pub fn add_regions_vec(mut self, regions: Vec<Region<'c>>) -> Self {
        unsafe {
            // This may fire with -D clippy::nursery, however, it is
            // guaranteed by the std that ManuallyDrop<T> has the same layout as T
            #[allow(clippy::transmute_undefined_repr)]
            mlirOperationStateAddOwnedRegions(
                &mut self.raw,
                regions.len() as isize,
                transmute::<Vec<Region>, Vec<ManuallyDrop<Region>>>(regions).as_ptr() as *const _,
            )
        }

        self
    }

    /// Adds successor blocks.
    // TODO Fix this to ensure blocks are alive while they are referenced by the
    // operation.
    pub fn add_successors(mut self, successors: &[&Block<'c>]) -> Self {
        for block in successors {
            unsafe {
                mlirOperationStateAddSuccessors(&mut self.raw, 1, &[block.to_raw()] as *const _)
            }
        }

        self
    }

    /// Adds attributes.
    pub fn add_attributes(mut self, attributes: &[(Identifier<'c>, Attribute<'c>)]) -> Self {
        for (identifier, attribute) in attributes {
            unsafe {
                mlirOperationStateAddAttributes(
                    &mut self.raw,
                    1,
                    &[mlirNamedAttributeGet(
                        identifier.to_raw(),
                        attribute.to_raw(),
                    )] as *const _,
                )
            }
        }

        self
    }

    /// Enables result type inference.
    pub fn enable_result_type_inference(mut self) -> Self {
        unsafe { mlirOperationStateEnableResultTypeInference(&mut self.raw) }

        self
    }

    /// Builds an operation.
    pub fn build(mut self) -> Result<Operation<'c>, Error> {
        unsafe { Operation::from_option_raw(mlirOperationCreate(&mut self.raw)) }
            .ok_or(Error::OperationBuild)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{block::BlockLike, operation::OperationLike, Block, ValueLike},
        test::create_test_context,
    };

    #[test]
    fn new() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        OperationBuilder::new("foo", Location::unknown(&context))
            .build()
            .unwrap();
    }

    #[test]
    fn add_operands() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        let location = Location::unknown(&context);
        let r#type = Type::index(&context);
        let block = Block::new(&[(r#type, location)]);
        let argument = block.argument(0).unwrap().into();

        OperationBuilder::new("foo", Location::unknown(&context))
            .add_operands(&[argument])
            .build()
            .unwrap();
    }

    #[test]
    fn add_results() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        OperationBuilder::new("foo", Location::unknown(&context))
            .add_results(&[Type::parse(&context, "i1").unwrap()])
            .build()
            .unwrap();
    }

    #[test]
    fn add_regions() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        OperationBuilder::new("foo", Location::unknown(&context))
            .add_regions([Region::new()])
            .build()
            .unwrap();
    }

    #[test]
    fn add_successors() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        OperationBuilder::new("foo", Location::unknown(&context))
            .add_successors(&[&Block::new(&[])])
            .build()
            .unwrap();
    }

    #[test]
    fn add_attributes() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        OperationBuilder::new("foo", Location::unknown(&context))
            .add_attributes(&[(
                Identifier::new(&context, "foo"),
                Attribute::parse(&context, "unit").unwrap(),
            )])
            .build()
            .unwrap();
    }

    #[test]
    fn enable_result_type_inference() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        let location = Location::unknown(&context);
        let r#type = Type::index(&context);
        let block = Block::new(&[(r#type, location)]);
        let argument = block.argument(0).unwrap().into();

        assert_eq!(
            OperationBuilder::new("arith.addi", location)
                .add_operands(&[argument, argument])
                .enable_result_type_inference()
                .build()
                .unwrap()
                .result(0)
                .unwrap()
                .r#type(),
            r#type,
        );
    }

    #[test]
    fn add_operands_with_segment_sizes() {
        let context = create_test_context();
        context.set_allow_unregistered_dialects(true);

        let location = Location::unknown(&context);
        let r#type = Type::index(&context);
        let block = Block::new(&[(r#type, location), (r#type, location), (r#type, location)]);
        let arg0: Value = block.argument(0).unwrap().into();
        let arg1: Value = block.argument(1).unwrap().into();
        let arg2: Value = block.argument(2).unwrap().into();

        // Test with various segment sizes: 0, 1, 2
        let op = OperationBuilder::new("test.variadic_op", location)
            .add_operands_with_segment_sizes(
                &context,
                &[
                    &[],           // segment 0: empty
                    &[arg0],       // segment 1: one operand
                    &[arg1, arg2], // segment 2: two operands
                ],
            )
            .build()
            .unwrap();

        // Verify operand count
        assert_eq!(op.operand_count(), 3);

        // Verify the operandSegmentSizes attribute was added
        let attr = op.attribute("operandSegmentSizes").unwrap();
        let attr_str = attr.to_string();
        // DenseI32Array format is "array<i32: 0, 1, 2>"
        assert!(
            attr_str.contains("0") && attr_str.contains("1") && attr_str.contains("2"),
            "Expected segment sizes in attribute, got: {}",
            attr_str
        );
    }
}
