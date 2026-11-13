"""Central paths and constants for the data pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_REPORTS = ROOT / "data" / "reports"

RAW_MOVIES = DATA_RAW / "movies.csv"
RAW_RATINGS = DATA_RAW / "ratings.csv"
RAW_GENRES = DATA_RAW / "genres.json"

MOVIES_CLEAN = DATA_PROCESSED / "movies_clean.csv"
USERS_CLEAN = DATA_PROCESSED / "users.csv"
RATINGS_LONG = DATA_PROCESSED / "ratings_long.csv"
RATINGS_TRAIN = DATA_PROCESSED / "ratings_train.csv"
RATINGS_TEST = DATA_PROCESSED / "ratings_test.csv"
MOVIE_POPULARITY = DATA_PROCESSED / "movie_popularity_train.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
RATING_MIN = 1.0
RATING_MAX = 5.0
LIKE_THRESHOLD = 3.0

# The 15 movies used in the survey, in the exact order of the rating columns
# in ratings.csv (cols 6..20). (movie_id, english title used as a sanity key).
SURVEY_MOVIES = [
    (299534, "Avengers: Endgame"),
    (597, "Titanic"),
    (138843, "The Conjuring"),
    (109445, "Frozen"),
    (168259, "Fast & Furious 7"),
    (296098, "Me Before You"),
    (157336, "Interstellar"),
    (18785, "The Hangover"),
    (671, "Harry Potter and the Philosopher's Stone"),
    (496243, "Parasite"),
    (372058, "Your Name"),
    (396535, "Train to Busan"),
    (354912, "Coco"),
    (18886, "Shutter"),
    (1083862, "Hunger"),
]
