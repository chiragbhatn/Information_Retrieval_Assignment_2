"""Static word lists, bundled so the app works without downloading NLTK corpora."""

ENGLISH_STOPWORDS = frozenset("""
a about above after again against ain all am an and any are aren as at be because been before being
below between both but by can could couldn d did didn do does doesn doing don down during each few for
from further had hadn has hasn have haven having he her here hers herself him himself his how i if in
into is isn it its itself just ll m ma me might mightn more most must mustn my myself needn no nor not
now o of off on once only or other our ours ourselves out over own re s same shan she should shouldn
so some such t than that the their theirs them themselves then there these they this those through to
too under until up us ve very was wasn we were weren what when where which while who whom why will
with won would wouldn y you your yours yourself yourselves also may shall upon via yet
""".split())

# Words that are frequent in any news story and carry little topical meaning.
NEWS_STOPWORDS = frozenset("""
said says say saying told mr mrs ms bbc news year years new one two three first last people time
like get got make made way well back going go see week could would also many much
""".split())
