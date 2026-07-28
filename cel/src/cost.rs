use crate::{ExecutionError, Value};
use std::collections::HashMap;
use std::sync::Arc;

/// Estimates the runtime cost of a user-defined function call.
pub trait ActualCostEstimator: Send + Sync {
    fn call_cost(
        &self,
        function: &str,
        overload_id: Option<&str>,
        args: &[Value],
        result: &Value,
    ) -> Option<u64>;
}

/// Computes the runtime cost of a specific function overload.
pub trait FunctionCostTracker: Send + Sync {
    fn call_cost(&self, args: &[Value], result: &Value) -> Option<u64>;
}

impl<F> FunctionCostTracker for F
where
    F: Fn(&[Value], &Value) -> Option<u64> + Send + Sync,
{
    fn call_cost(&self, args: &[Value], result: &Value) -> Option<u64> {
        self(args, result)
    }
}

/// Runtime cost accounting state for one CEL evaluation.
pub struct CostTracker {
    estimator: Option<Arc<dyn ActualCostEstimator>>,
    overload_trackers: HashMap<String, Arc<dyn FunctionCostTracker>>,
    limit: Option<u64>,
    actual_cost: u64,
    presence_test_has_cost: bool,
}

impl Default for CostTracker {
    fn default() -> Self {
        Self {
            estimator: None,
            overload_trackers: HashMap::new(),
            limit: None,
            actual_cost: 0,
            presence_test_has_cost: true,
        }
    }
}

impl CostTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_limit(limit: u64) -> Self {
        Self {
            limit: Some(limit),
            ..Self::default()
        }
    }

    pub fn set_limit(&mut self, limit: Option<u64>) {
        self.limit = limit;
    }

    pub fn limit(&self) -> Option<u64> {
        self.limit
    }

    pub fn actual_cost(&self) -> u64 {
        self.actual_cost
    }

    pub fn presence_test_has_cost(&self) -> bool {
        self.presence_test_has_cost
    }

    pub fn set_presence_test_has_cost(&mut self, has_cost: bool) {
        self.presence_test_has_cost = has_cost;
    }

    pub fn set_estimator(&mut self, estimator: Arc<dyn ActualCostEstimator>) {
        self.estimator = Some(estimator);
    }

    pub fn register_overload_tracker(
        &mut self,
        overload_id: impl Into<String>,
        tracker: Arc<dyn FunctionCostTracker>,
    ) {
        self.overload_trackers.insert(overload_id.into(), tracker);
    }

    pub(crate) fn charge(&mut self, amount: u64) -> Result<(), ExecutionError> {
        self.actual_cost = self.actual_cost.saturating_add(amount);
        if let Some(limit) = self.limit {
            if self.actual_cost > limit {
                return Err(ExecutionError::CostLimitExceeded {
                    limit,
                    actual: self.actual_cost,
                });
            }
        }
        Ok(())
    }

    pub(crate) fn charge_call(
        &mut self,
        function: &str,
        overload_id: Option<&str>,
        args: &[Value],
        result: &Value,
    ) -> Result<(), ExecutionError> {
        if let Some(overload_id) = overload_id {
            if let Some(tracker) = self.overload_trackers.get(overload_id) {
                if let Some(cost) = tracker.call_cost(args, result) {
                    return self.charge(cost);
                }
            }
        }

        if let Some(estimator) = &self.estimator {
            if let Some(cost) = estimator.call_cost(function, overload_id, args, result) {
                return self.charge(cost);
            }
        }

        self.charge(default_call_cost(function, args))
    }
}

/// Result of a cost-tracked CEL execution.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionResult {
    pub value: Value,
    pub actual_cost: u64,
}

pub(crate) fn actual_size(value: &Value) -> u64 {
    match value {
        Value::String(value) => value.len() as u64,
        Value::Bytes(value) => value.len() as u64,
        Value::List(value) => value.len() as u64,
        Value::Map(value) => value.map.len() as u64,
        Value::Opaque(value) => value
            .downcast_ref::<crate::objects::OptionalValue>()
            .and_then(|optional| optional.value())
            .map(actual_size)
            .unwrap_or(1),
        _ => 1,
    }
}

fn traversal_cost(size: u64) -> u64 {
    // CEL-Go uses a fractional traversal factor. Keeping the helper isolated
    // makes it straightforward to align the constant with checker cost values.
    size.max(1)
}

fn default_call_cost(function: &str, args: &[Value]) -> u64 {
    match function {
        "startsWith" | "endsWith" | "string" | "bytes" => args
            .first()
            .map(actual_size)
            .map(traversal_cost)
            .unwrap_or(1),
        "contains" if args.len() >= 2 => {
            traversal_cost(actual_size(&args[0]))
                .saturating_mul(traversal_cost(actual_size(&args[1])))
        }
        "matches" if args.len() >= 2 => {
            let string_cost = traversal_cost(actual_size(&args[0]).saturating_add(1));
            let regex_cost = traversal_cost(actual_size(&args[1]).div_ceil(4));
            string_cost.saturating_mul(regex_cost)
        }
        "_in_" if args.len() >= 2 => actual_size(&args[1]),
        "_+_" if args.len() >= 2 => {
            traversal_cost(actual_size(&args[0]).saturating_add(actual_size(&args[1])))
        }
        "_==_" | "_!=_" | "_<_" | "_<=_" | "_>_" | "_>=_" if args.len() >= 2 => {
            traversal_cost(actual_size(&args[0]).min(actual_size(&args[1])))
        }
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_runtime_values() {
        assert_eq!(actual_size(&Value::from("hello")), 5);
        assert_eq!(actual_size(&Value::from(vec![1_i64, 2, 3])), 3);
        assert_eq!(actual_size(&Value::from(42_i64)), 1);
    }

    #[test]
    fn rejects_cost_above_limit() {
        let mut tracker = CostTracker::with_limit(2);
        tracker.charge(2).unwrap();
        assert_eq!(
            tracker.charge(1),
            Err(ExecutionError::CostLimitExceeded {
                limit: 2,
                actual: 3,
            })
        );
    }

    #[test]
    fn string_contains_cost_is_product() {
        assert_eq!(
            default_call_cost(
                "contains",
                &[Value::from("abcdef"), Value::from("bc")],
            ),
            12
        );
    }
}
