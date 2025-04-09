from rtamt import StlDiscreteTimeSpecification
from typing import Dict, List, Tuple, Optional


class STLProperty:
    """Base class for STL formal properties"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.spec = None
        
    def initialize(self):
        """Initialize the STL specification"""
        self.spec = StlDiscreteTimeSpecification()
        self._declare_variables()
        self._define_specification()
        self.spec.parse()
        
    def _declare_variables(self):
        """Declare variables used in the specification"""
        raise NotImplementedError
        
    def _define_specification(self):
        """Define the STL specification formula"""
        raise NotImplementedError
        
    def evaluate(self, signals: Dict[str, List]) -> Tuple[float, List[Tuple[int, float]]]:
        """
        Evaluate the property on the provided signals
        
        Args:
            signals: Dictionary mapping variable names to signal values
            
        Returns:
            Tuple of (minimum_robustness, full_robustness_trace)
        """
        if self.spec is None:
            self.initialize()
            
        robustness_trace = self.spec.evaluate(signals)
        min_robustness = min(robustness_trace, key=lambda x: x[1])[1]
        
        return min_robustness, robustness_trace
        
    def is_satisfied(self, signals: Dict[str, List]) -> bool:
        """Check if the property is satisfied by the signals"""
        min_robustness, _ = self.evaluate(signals)
        return min_robustness >= 0


class LogitCosineSimilarityProperty(STLProperty):
    """Property ensuring that logits maintain cosine similarity above threshold"""
    
    def __init__(self, threshold: float = 0.85):
        super().__init__(
            name="logit_cosine_similarity",
            description=f"Model output logits maintain cosine similarity above {threshold}"
        )
        self.threshold = threshold
        
    def _declare_variables(self):
        self.spec.declare_var('cos_sim', 'float')
        
    def _define_specification(self):
        self.spec.spec = f'always (cos_sim >= {self.threshold})'


class MaxProbabilityDeviationProperty(STLProperty):
    """Property ensuring that output probabilities don't deviate too much"""
    
    def __init__(self, max_deviation: float = 0.1):
        super().__init__(
            name="max_probability_deviation",
            description=f"Output probabilities don't deviate more than {max_deviation}"
        )
        self.max_deviation = max_deviation
        
    def _declare_variables(self):
        self.spec.declare_var('prob_diff', 'float')
        
    def _define_specification(self):
        self.spec.spec = f'always (prob_diff <= {self.max_deviation})'


class TopKOverlapProperty(STLProperty):
    """Property ensuring that top-k predictions maintain sufficient overlap"""
    
    def __init__(self, k: int = 5, min_overlap: float = 0.6):
        super().__init__(
            name="topk_overlap",
            description=f"Top-{k} predictions maintain at least {min_overlap*100}% overlap"
        )
        self.k = k
        self.min_overlap = min_overlap
        
    def _declare_variables(self):
        self.spec.declare_var('overlap_ratio', 'float')
        
    def _define_specification(self):
        self.spec.spec = f'always (overlap_ratio >= {self.min_overlap})'


class ResponseTimeProperty(STLProperty):
    """Property ensuring that model response time stays under threshold"""
    
    def __init__(self, max_time_ratio: float = 1.5):
        super().__init__(
            name="response_time",
            description=f"Model response time stays under {max_time_ratio}x the original"
        )
        self.max_time_ratio = max_time_ratio
        
    def _declare_variables(self):
        self.spec.declare_var('time_ratio', 'float')
        
    def _define_specification(self):
        self.spec.spec = f'always (time_ratio <= {self.max_time_ratio})'


# Registry of available properties
PROPERTY_REGISTRY = {
    "cosine_similarity": LogitCosineSimilarityProperty,
    "probability_deviation": MaxProbabilityDeviationProperty,
    "topk_overlap": TopKOverlapProperty,
    "response_time": ResponseTimeProperty
}


def get_property(name: str, **kwargs) -> STLProperty:
    """Get a property by name with optional configuration"""
    if name not in PROPERTY_REGISTRY:
        raise ValueError(f"Unknown property: {name}. Available properties: {list(PROPERTY_REGISTRY.keys())}")
        
    return PROPERTY_REGISTRY[name](**kwargs)