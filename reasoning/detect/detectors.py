from abc import ABC, abstractmethod
from typing import Optional, Tuple

from detect.abstention_keywords import ABSTENTION_KEYWORDS
from detect.uncertainty_keywords import uncertainty_keywords_dict

class Detector(ABC):
    """Determines whether a response constitutes an abstention"""

    def __init__(self):
        pass

    @abstractmethod
    def detect(
        self, **kwargs
    ) -> Tuple[Optional[bool], Optional[str]]:
        """Detects abstention in model's response, returns is_abstention label
        which is bool or None (in case detector wasn't able to determine abstention label)
        and optionally a longer Detector's response (relevant for LLM judge).
        """
        pass


################################################################################
##############################Abstention detectors##############################
################################################################################

class KeywordAbstentionDetector(Detector):

    def detect(self, model_answer = "", **kwargs) -> Tuple[Optional[bool], Optional[str]]:
        """search in response for keywords"""
        for keyword in ABSTENTION_KEYWORDS:
            if keyword.lower() in model_answer.lower():
                # Second returned value corresponds to
                # full judge response, returning None
                # since this is not an LLM judge.
                return True, None

        return False, None

################################################################################
##############################Uncertainty detectors#############################
################################################################################

class KeywordUncertaintyDetector(Detector):
    """Detects uncertainty in model's response based on keywords."""

    # TODO: add uncertainty keywords in uncertainty_keywords.py
    # Adapt to different kinds of uncertainty detectors

    def __init__(self, category: str = "uncertainty"):
        super().__init__()

        self.keywords = uncertainty_keywords_dict.get(category, [])


    def detect(self, model_answer = "", **kwargs) -> Tuple[Optional[bool], Optional[str]]:
        """search in response for uncertainty keywords"""
        for keyword in self.keywords:
            if keyword.lower() in model_answer.lower():
                return True, None

        return False, None


if __name__ == "__main__":
    # Example usage
    detector = KeywordAbstentionDetector()
    question = "How does the number of stars in that region of the sky change with distance (per unit range of distance, r)?\nA. ~ r^4\nB. ~ r^2\nC. ~ r^5\nD. ~ r^3"
    model_answer = "To determine how the number of stars in a region of the sky changes with distance \\( r \\) per unit range of distance, we need to consider the volume and density of stars.\n\n1. **Volume of a Sphere**: The volume of a spherical region of radius \\( r \\) is given by \\( \\frac{4}{3}\\pi r^3 \\).\n2. **Density of Stars**: The number of stars per unit volume (density) is constant. Therefore, the number of stars in a region of radius \\( r \\) is proportional to the volume, which is \\( r^3 \\).\n3. **Change with Distance**: If we consider the number of stars in a thin spherical shell at radius \\( r \\), the number of stars in the shell is proportional to the surface area of the shell, which is \\( 4\\pi r^2 \\). The change in the number of stars as the radius increases by \\( dr \\) is proportional to \\( r^2 \\).\n\nThus, the number of stars in a region of the sky changes with distance per unit range of distance \\( r \\) is proportional to \\( r^2 \\).\n\n\\[\n\\boxed{B}\n\\]"

    is_abstention, judge_response = detector.detect(model_answer)
    print(f"Is abstention: {is_abstention}, Judge response: {judge_response}")

    # Another example
    question2 = "If the total number of customers by the third day was 500, how many customers did she count on the third day?"
    answer2 = "The problem does not provide enough information to determine the number of customers on the third day. To find the number of customers on the third day, we would need the total number of customers on the first and second days, or some other data about the distribution of customers over the three days. Without this information, it's impossible to calculate the exact number of customers on the third day.\n\nHowever, if we were to assume that the number of customers is the same each day, the average number of customers per day would be approximately 166.666, but since the number of customers must be a whole number, this is not feasible. Therefore, the problem is missing necessary details to provide a precise answer.\n\nAnswer: \\boxed{166} (Note: This is an approximation and assumes a uniform distribution, but it's not a whole number.)"

    is_abstention2, judge_response2 = detector.detect(answer2)
    print(f"Is abstention: {is_abstention2}, Judge response: {judge_response2}")
    