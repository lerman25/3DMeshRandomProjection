import sklearn; sklearn.show_versions()
class PitchConstants:
    #contants
    #Size based on FIFA's standart (https://publications.fifa.com/fr/football-stadiums-guidelines/technical-guideline/stadium-guidelines/pitch-dimensions-and-surrounding-areas/)
    FIELD_LENGTH = 105 # meter
    FIELD_WIDTH = 68 # meter
    MODEL_HEIGHT = 10 # meter (Max limit)
    CENTER_CIRCLE_RADIUS = 9.15 # meter
    PENALTY_AREA_LENGTH = 16.5 # meter
    PENALTY_AREA_WIDTH = 40.32 # meter
    GOAL_AREA_WIDTH = 18.32 # meter
    GOAL_AREA_HEIGHT = 5.5 # meter
    PENALTY_SPOT_DISTANCE = 11 # meter
    POSTS_WIDTH = 0.12 # meter
    GOAL_DEPTH = 1.5 # meter
    GOAL_WIDTH = 7.32 # meter
    GOAL_HEIGHT = 2.44 # meter
    CORNER_ARC_RADIUS = 1 # meter
    PENALTY_ARC_RADIUS = 9.15 # meter
    #Line thickness based on FA's standart (https://www.thefa.com/-/media/files/thefaportal/governance-docs/rules-of-the-association/2021-22/goalpost-and-pitch-sizes-and-line-marking.ashx)
    LINE_THICKNESS = POSTS_WIDTH 
    def __init__(self):
    # Set constants from separate classes as attributes
        for key, value in PitchConstants.__dict__.items():
            if not key.startswith("__"):
                self.__dict__.update(**{key: value})

    def __setattr__(self, name, value):
        raise TypeError("Constants are immutable")
