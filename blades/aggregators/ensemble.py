from typing import List, Dict, Any, Optional, Type
import importlib
import sys
import torch


class EnsembleAggregator(object):
    """Aggregator that combines multiple aggregators by averaging their results.
    
    This aggregator takes multiple aggregation methods by their class names,
    initializes them with provided parameters, and applies each one
    to the input tensors, then averages the results to produce the final output.
    
    Example:
        >>> aggregators = [
        >>>     {"name": "Mean", "params": {}},
        >>>     {"name": "Median", "params": {}},
        >>>     {"name": "Trimmedmean", "params": {"num_byzantine": 2}}
        >>> ]
        >>> ensemble = EnsembleAggregator(aggregators)
        >>> result = ensemble(inputs)
    """
    
    def __init__(self, num_byzantine, aggregators_config: List[Dict[str, Any]]):
        """Initialize the ensemble aggregator with a list of aggregator configurations.
        
        Args:
            aggregators_config: List of dictionaries with keys:
                - "name": String name of the aggregator class
                - "params": Dictionary of parameters to pass to the aggregator constructor
        """
        if not aggregators_config:
            raise ValueError("At least one aggregator must be provided.")
        
        self.aggregators = []
        
        # Get the current module
        current_module = sys.modules[__name__]
        
        # Find the parent module where aggregator classes are defined
        # This assumes EnsembleAggregator is in the same module as other aggregators
        parent_module = importlib.import_module(current_module.__package__) if hasattr(current_module, "__package__") else current_module
        
        # Initialize each aggregator
        for agg_config in aggregators_config:
            name = agg_config["name"]
            params = agg_config.get("params", {})
            
            if ("DnC" in name) \
                or ("Trimmedmean" in name) \
                or ("Multikrum" in name):
                params['num_byzantine'] = num_byzantine
            
            # Get the aggregator class
            agg_class = getattr(parent_module, name)
            
            # Initialize the aggregator with parameters
            agg_instance = agg_class(**params)
            
            self.aggregators.append(agg_instance)
    
    def __call__(self, inputs: List[torch.Tensor]):
        """Apply all aggregators and average their results.
        
        Args:
            inputs: List of torch.Tensor objects to be aggregated.
            
        Returns:
            torch.Tensor: The averaged result of all aggregators.
        """
        # Apply each aggregator to the inputs
        results = [agg(inputs) for agg in self.aggregators]
        
        # Average the results
        ensemble_result = torch.stack(results).mean(dim=0)
        
        return ensemble_result