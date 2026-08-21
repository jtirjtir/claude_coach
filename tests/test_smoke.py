def test_imports(bot):
    assert bot.get_goal()["name"]


def test_default_goal_is_dateless(bot):
    """A dated default meant a wiped state.json silently resurrected that race,
    and kept briefing for it long after the date passed."""
    assert bot.DEFAULT_GOAL["date"] is None
    assert bot.has_goal(bot.DEFAULT_GOAL) is False
    assert bot.race_date(bot.DEFAULT_GOAL) is None
    assert bot.days_to_goal(goal=bot.DEFAULT_GOAL) is None
    assert bot.goal_is_past(bot.DEFAULT_GOAL) is False   # unset is not "passed"
