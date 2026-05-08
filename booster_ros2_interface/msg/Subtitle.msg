string magic_number
string text
string language
string user_id    # Indicates the source of the subtitle. If it's from the robot, it will be the fixed value "BoosterRobot". Otherwise, it represents a human and may be a random string.
int32 seq         # subtitle sequence
bool definite     # Indicates whether the subtitle is a complete sentence.
bool paragraph    # Indicates whether the subtitle is a complete paragraph.
int32 round_id
