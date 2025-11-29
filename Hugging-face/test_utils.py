import unittest
from utils import map_sound_to_vibration, clean_text
from mcp_server import MCPServer

class TestAccessibilityUtils(unittest.TestCase):
    def test_haptics_mapping(self):
        self.assertEqual(map_sound_to_vibration("dog_bark")["pattern"], "short-short")
        self.assertEqual(map_sound_to_vibration("car_horn")["pattern"], "strong-pulse")
        self.assertEqual(map_sound_to_vibration("unknown_sound")["pattern"], "default")

    def test_clean_text(self):
        self.assertEqual(clean_text("  Hello   World  "), "Hello World")

    def test_mcp_server(self):
        server = MCPServer()
        self.assertIn("Calendar", server.get_calendar_events())
        self.assertIn("emails", server.summarize_emails())

if __name__ == "__main__":
    unittest.main()
