def test_imports(bot):
    assert bot.get_goal()["name"]
    assert bot.race_date() is not None
