"""Stand-in for the original Django models: only the Tag type constants are used by the pipeline."""
class Tag:
    TAG_TYPE_BINARY = 'BINARY'
    TAG_TYPE_CONTINUOUS = 'CONTINUOUS'
    TAG_TYPE_POSITION = 'POSITION'
    TAG_TYPE_ACCUMULATE = 'ACCUMLATE'
    TAG_TYPE_GATE = 'GATE'
    TAG_TYPE_PERCENT = 'PERCENT'
    TAG_TYPE_COUNTER = 'COUNTER'
    TAG_TYPE_OTHERS = 'OTHERS'
    TAG_TYPE_ZERO = 'ZERO'

class TagValue:
    pass

class TagEvents:
    pass
