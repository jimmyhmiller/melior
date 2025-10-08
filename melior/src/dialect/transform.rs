//! `transform` dialect.

use crate::{
    ir::{operation::OperationLike, Operation},
    logical_result::LogicalResult,
    Error,
};
use mlir_sys::{
    mlirMergeSymbolsIntoFromClone, mlirTransformApplyNamedSequence, mlirTransformOptionsCreate,
    mlirTransformOptionsDestroy, mlirTransformOptionsEnableExpensiveChecks,
    mlirTransformOptionsEnforceSingleTopLevelTransformOp,
    mlirTransformOptionsGetEnforceSingleTopLevelTransformOp,
    mlirTransformOptionsGetExpensiveChecksEnabled, MlirTransformOptions,
};

/// Transform options for configuring transform dialect operations.
#[derive(Debug)]
pub struct TransformOptions {
    raw: MlirTransformOptions,
}

impl TransformOptions {
    /// Creates a new transform options object with default configuration.
    pub fn new() -> Self {
        Self {
            raw: unsafe { mlirTransformOptionsCreate() },
        }
    }

    /// Enables or disables expensive checks in transform operations.
    pub fn enable_expensive_checks(&self, enable: bool) -> &Self {
        unsafe { mlirTransformOptionsEnableExpensiveChecks(self.raw, enable) };
        self
    }

    /// Returns whether expensive checks are enabled.
    pub fn expensive_checks_enabled(&self) -> bool {
        unsafe { mlirTransformOptionsGetExpensiveChecksEnabled(self.raw) }
    }

    /// Enables or disables enforcement of single top-level transform operation.
    pub fn enforce_single_top_level_transform_op(&self, enable: bool) -> &Self {
        unsafe { mlirTransformOptionsEnforceSingleTopLevelTransformOp(self.raw, enable) };
        self
    }

    /// Returns whether single top-level transform operation enforcement is enabled.
    pub fn single_top_level_transform_op_enforced(&self) -> bool {
        unsafe { mlirTransformOptionsGetEnforceSingleTopLevelTransformOp(self.raw) }
    }

    /// Converts transform options to a raw object.
    pub fn to_raw(&self) -> MlirTransformOptions {
        self.raw
    }
}

impl Default for TransformOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for TransformOptions {
    fn drop(&mut self) {
        unsafe { mlirTransformOptionsDestroy(self.raw) };
    }
}

/// Applies a transform script to a payload operation.
///
/// This function applies the transformation sequence defined in `transform_root` to the
/// `payload` operation, using the symbols defined in `transform_module`.
///
/// # Arguments
/// * `payload` - The operation to be transformed
/// * `transform_root` - The root operation of the transform sequence
/// * `transform_module` - The module containing transform symbol definitions
/// * `transform_options` - Configuration options for the transformation
///
/// # Returns
/// A `LogicalResult` indicating success or failure of the transformation.
pub fn apply_named_sequence(
    payload: &Operation,
    transform_root: &Operation,
    transform_module: &Operation,
    transform_options: &TransformOptions,
) -> Result<(), Error> {
    let result = unsafe {
        LogicalResult::from_raw(mlirTransformApplyNamedSequence(
            payload.to_raw(),
            transform_root.to_raw(),
            transform_module.to_raw(),
            transform_options.to_raw(),
        ))
    };

    if result.is_success() {
        Ok(())
    } else {
        Err(Error::OperationBuild) // Using existing error type, could add a more specific one
    }
}

/// Merges symbols from one operation into another.
///
/// This function merges symbols from the `other` operation into the `target` operation,
/// potentially renaming symbols to avoid conflicts.
///
/// # Arguments
/// * `target` - The target operation that will receive the merged symbols
/// * `other` - The source operation containing symbols to be merged
///
/// # Returns
/// A `LogicalResult` indicating success or failure of the merge operation.
pub fn merge_symbols_into_from_clone(target: &Operation, other: &Operation) -> Result<(), Error> {
    let result = unsafe {
        LogicalResult::from_raw(mlirMergeSymbolsIntoFromClone(
            target.to_raw(),
            other.to_raw(),
        ))
    };

    if result.is_success() {
        Ok(())
    } else {
        Err(Error::OperationBuild) // Using existing error type, could add a more specific one
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        context::Context,
        dialect::DialectHandle,
        ir::{Location, Module},
        test::load_all_dialects,
    };

    #[test]
    fn transform_options_new() {
        let options = TransformOptions::new();
        // Just test that the options can be created and queried without assuming default values
        let _expensive_checks = options.expensive_checks_enabled();
        let _single_top_level = options.single_top_level_transform_op_enforced();
    }

    #[test]
    fn transform_options_enable_expensive_checks() {
        let options = TransformOptions::new();

        // Test setting to true
        options.enable_expensive_checks(true);
        assert!(options.expensive_checks_enabled());

        // Test setting to false
        options.enable_expensive_checks(false);
        assert!(!options.expensive_checks_enabled());
    }

    #[test]
    fn transform_options_enforce_single_top_level() {
        let options = TransformOptions::new();

        // Test setting to true
        options.enforce_single_top_level_transform_op(true);
        assert!(options.single_top_level_transform_op_enforced());

        // Test setting to false
        options.enforce_single_top_level_transform_op(false);
        assert!(!options.single_top_level_transform_op_enforced());
    }

    #[test]
    fn transform_dialect_handle() {
        let context = Context::new();
        DialectHandle::transform().load_dialect(&context);
    }

    #[test]
    fn merge_symbols_test() {
        let context = Context::new();
        load_all_dialects(&context);
        DialectHandle::transform().load_dialect(&context);

        let location = Location::unknown(&context);
        let module1 = Module::new(location);
        let module2 = Module::new(location);

        // This should not fail with empty modules
        merge_symbols_into_from_clone(&module1.as_operation(), &module2.as_operation()).unwrap();
    }
}
