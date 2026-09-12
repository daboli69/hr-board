import datetime as dt
import unittest
from season_refresh import should_refresh

class SeasonRefreshTests(unittest.TestCase):
    def test_active_and_offseason(self):
        monday=dt.datetime(2026,12,7,10,30,tzinfo=dt.timezone.utc)
        self.assertTrue(should_refresh(1,'weekly',monday))
        self.assertTrue(should_refresh(0,'weekly',monday))
        self.assertFalse(should_refresh(0,'active',monday))
        self.assertFalse(should_refresh(0,'weekly',monday+dt.timedelta(hours=2)))
        self.assertFalse(should_refresh(0,'weekly',monday+dt.timedelta(days=1)))
if __name__=='__main__':unittest.main()
