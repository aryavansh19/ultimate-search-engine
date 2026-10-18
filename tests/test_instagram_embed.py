from __future__ import annotations

import json
import unittest

from extractor.envelope import MediaKind
from extractor.extractors.opengraph import _parse_instagram_embed


class InstagramEmbedTests(unittest.TestCase):
    def test_extracts_public_reel_context(self) -> None:
        context = {
            "gql_data": {
                "shortcode_media": {
                    "is_video": True,
                    "display_url": "https://cdn.example/cover.jpg",
                    "video_url": "https://cdn.example/reel.mp4?signature=temporary",
                    "video_duration": 134.5,
                    "video_view_count": 297_633,
                    "edge_liked_by": {"count": 28_098},
                    "edge_media_to_comment": {"count": 113},
                    "edge_media_to_caption": {
                        "edges": [
                            {
                                "node": {
                                    "text": "Five Git concepts that prevent mistakes. #git #devops"
                                }
                            }
                        ]
                    },
                    "owner": {"username": "_devoops"},
                }
            }
        }
        html = (
            '<script>init({"contextJSON":'
            + json.dumps(json.dumps(context))
            + "});</script>"
        )

        fields = _parse_instagram_embed(html)

        self.assertEqual(fields["title"], "_devoops on Instagram")
        self.assertEqual(fields["author"], "_devoops")
        self.assertEqual(fields["media_kind"], MediaKind.VIDEO)
        self.assertEqual(fields["duration_s"], 134.5)
        self.assertEqual(fields["view_count"], 297_633)
        self.assertEqual(fields["like_count"], 28_098)
        self.assertEqual(fields["comment_count"], 113)
        self.assertEqual(fields["platform_tags"], ["git", "devops"])
        self.assertIn("Five Git concepts", fields["caption"])

    def test_ignores_malformed_context(self) -> None:
        self.assertEqual(_parse_instagram_embed('{"contextJSON":"not-json"}'), {})


if __name__ == "__main__":
    unittest.main()
