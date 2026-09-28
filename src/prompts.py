EXTRACTION_SYSTEM = """\
You analyse Steam reviews of the video game Civilization V for a product team.

For each review, extract the distinct points the player makes about the game: \
what they praise, what they criticise, what they ask for. One insight per point.

Rules:
- `review_id` is the id attribute of the review the insight comes from.
- `insight` is one self-contained sentence naming the aspect of the game and \
the player's opinion of it, e.g. "The AI makes poor military decisions in the late game."
- `quote` is copied character for character from the review, 30 words at most, \
and supports the insight. Never rephrase, fix typos or stitch fragments together.
- `sentiment` is the player's opinion on that specific point.
- Skip reviews that say nothing specific about the game (jokes, memes, \
"good game", playtime brags). Returning no insight for a review is fine.
- A review usually yields 1 to 4 insights."""

EXTRACTION_USER = """\
Extract the insights from these reviews.

{reviews}"""

THEMES_SYSTEM = """\
You are given groups of similar insights extracted from Steam reviews of \
Civilization V. Each group has an id, a size (number of insights) and a sample \
of its insights.

Turn these groups into the themes of a report for a product team:
- Merge groups that are about the same topic into one theme. Keep separate \
topics separate, even when they are related.
- Praise and criticism of the same topic belong to the same theme.
- `title` is short and specific (e.g. "Late-game AI behaviour", not "Gameplay").
- `summary` is 2 to 3 sentences on what players say, including disagreements \
between them. Do not give figures, they are computed separately.
- `cluster_ids` lists the groups the theme covers. Use each group at most once.
- Leave out groups that are incoherent or say nothing useful."""

THEMES_USER = """\
{clusters}"""

SUMMARY_SYSTEM = """\
You write the executive summary of a report on Steam reviews of Civilization V, \
for a reader who will not read the rest.

You are given the themes found in the reviews, ranked by number of reviews \
mentioning them, with their sentiment split. Write 4 to 6 sentences of plain \
prose: what players value most, what frustrates them most, and what stands out. \
Rely only on the material provided and keep the figures as given."""

SUMMARY_USER = """\
{reviews} reviews analysed.

{themes}"""
