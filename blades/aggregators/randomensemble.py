from typing import List, Dict, Any, Optional, Type
import importlib
import random
import sys
import torch


class RandomEnsembleAggregator(object):
    """Aggregator that combines multiple aggregators by averaging their results.
    
    This aggregator takes multiple aggregation methods by their class names,
    initializes them with provided parameters, and chooses a random one for every
    iteration to apply to all of the input tensors to produce the final output.
    
    Example:
        >>> aggregators = [
        >>>     {"name": "Mean", "params": {}},
        >>>     {"name": "Median", "params": {}},
        >>>     {"name": "Trimmedmean", "params": {"num_byzantine": 2}}
        >>> ]
        >>> ensemble = RandomEnsembleAggregator(aggregators)
        >>> result = ensemble(inputs)
    """
    
    def __init__(self, aggregators_config: List[Dict[str, Any]]):
        """Initialize the random ensemble aggregator with a list of aggregator configurations.
        
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
        # This assumes RandomEnsembleAggregator is in the same module as other aggregators
        parent_module = importlib.import_module(current_module.__package__) if hasattr(current_module, "__package__") else current_module
        
        # Initialize each aggregator
        for agg_config in aggregators_config:
            name = agg_config["name"]
            params = agg_config.get("params", {})
            
            # Get the aggregator class
            agg_class = getattr(parent_module, name)
            
            # Initialize the aggregator with parameters
            agg_instance = agg_class(**params)
            
            self.aggregators.append(agg_instance)
    
    def __call__(self, inputs: List[torch.Tensor]):
        """Apply a randomly chosen aggregators.
        
        Args:
            inputs: List of torch.Tensor objects to be aggregated.
            
        Returns:
            torch.Tensor: The result of a randomly chosen aggregator.
        """
        # Select an aggregator
        agg = random.choice(self.aggregators)
        
        # Apply the aggregator to the inputs
        return agg(inputs)