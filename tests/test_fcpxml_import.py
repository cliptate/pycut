import json
import xml.etree.ElementTree as ET


def test_cli_rough_cuts_fcpxml_from_transcript(tmp_path, monkeypatch):
    import pycut.cli as cli
    import pycut.translation as translation

    class FakeTranslator:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def translate(self, texts, src, dest):
            assert (src, dest) == ("en", "zh")
            return [type("Translation", (), {"text": f"tr:{text}"})() for text in texts]

    monkeypatch.setattr(translation, "Translator", FakeTranslator)

    source = tmp_path / "project.fcpxml"
    source.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<fcpxml version="1.11">
  <resources>
    <format id="r1" frameDuration="1/25s"/>
    <asset id="r2" name="Interview" start="0/25s" duration="250/25s" format="r1">
      <media-rep kind="original-media" src="file:///tmp/interview.mov"/>
    </asset>
    <effect id="r3" name="Existing Effect" uid="existing-effect"/>
  </resources>
  <library><event name="Event"><project name="Rough Cut">
    <sequence format="r1" duration="250/25s"><spine>
      <asset-clip ref="r2" name="Interview" offset="0/25s" duration="250/25s">
        <adjust-transform position="10 20"/>
        <filter-video ref="r3"/>
      </asset-clip>
    </spine></sequence>
    <metadata><md key="com.example.note" value="keep me"/></metadata>
  </project></event></library>
</fcpxml>
""",
        encoding="utf-8",
    )
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 1, "end": 2, "text": "first"},
                    {"start": 2, "end": 3, "text": ""},
                    {"start": 4, "end": 5, "text": "second"},
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"

    result = cli.main(
        [
            str(source),
            "--transcript",
            str(transcript),
            "--format",
            "fcpxml",
            "--translate",
            "--source-lang",
            "en",
            "--target-lang",
            "zh",
            "--orientation",
            "portrait",
            "--original-subtitle-color",
            "#123456",
            "--translation-subtitle-color",
            "#ABCDEF",
            "--margin-left",
            "0",
            "--margin-right",
            "0",
            "-o",
            str(output_dir),
        ]
    )

    output = output_dir / "project.fcpxml"
    root = ET.parse(output).getroot()
    clips = root.findall(".//spine/asset-clip")
    titles = root.findall(".//spine/asset-clip/title")
    assert (
        result["fcpxml"],
        root.find(".//project").attrib["name"],
        [(clip.attrib["offset"], clip.attrib["start"], clip.attrib["duration"]) for clip in clips],
        ["".join(run.text or "" for run in title.findall("./text/text-style")) for title in titles],
        [(title.attrib["ref"], title.attrib["offset"], title.attrib["duration"]) for title in titles],
        [
            (effect.attrib["id"], effect.attrib["name"], effect.attrib["uid"])
            for effect in root.findall("./resources/effect")
        ],
        [clip.find("filter-video").attrib["ref"] for clip in clips],
        [clip.find("adjust-transform").attrib["position"] for clip in clips],
        root.find('.//project/metadata/md[@key="com.example.note"]').attrib["value"],
    ) == (
        str(output),
        "Rough Cut",
        [
            ("0/25s", "25/25s", "25/25s"),
            ("25/25s", "100/25s", "25/25s"),
        ],
        ["first", "tr:first", "second", "tr:second"],
        [
            ("r4", "25/25s", "25/25s"),
            ("r4", "25/25s", "25/25s"),
            ("r4", "100/25s", "25/25s"),
            ("r4", "100/25s", "25/25s"),
        ],
        [
            ("r3", "Existing Effect", "existing-effect"),
            (
                "r4",
                "Subtitle",
                ".../Titles.localized/Subtitles.localized/Subtitle.localized/Subtitle.moti",
            ),
        ],
        ["r3", "r3"],
        ["10 20", "10 20"],
        "keep me",
    )
    assert [[child.tag for child in clip] for clip in clips] == [
        ["adjust-transform", "title", "title", "filter-video"],
        ["adjust-transform", "title", "title", "filter-video"],
    ]

    styles = root.findall(".//spine/asset-clip/title/text-style-def/text-style")
    assert [style.attrib["fontColor"] for style in styles] == [
        "0.0706 0.2039 0.3373 1",
        "0.6706 0.8039 0.9373 1",
        "0.0706 0.2039 0.3373 1",
        "0.6706 0.8039 0.9373 1",
    ]
    assert [style.attrib["fontSize"] for style in styles] == ["48", "48", "48", "48"]
    assert [style.attrib["lineSpacing"] for style in styles] == ["22", "22", "22", "22"]
    assert [title.attrib["lane"] for title in titles] == ["1", "2", "1", "2"]
    assert [title.attrib["role"] for title in titles] == ["subtitles.subtitles-1"] * 4
    assert [
        [(param.attrib["name"], param.attrib["key"], param.attrib["value"]) for param in title.findall("param")]
        for title in titles
    ] == [
        [
            ("Font Size", "9999/3336674837/3336674846/5/3336674848/3", "48"),
            ("Y Position Offset", "9999/3336678691/100/3337241559/2/100", "0.556301"),
        ],
        [
            ("Font Size", "9999/3336674837/3336674846/5/3336674848/3", "48"),
            ("Y Position Offset", "9999/3336678691/100/3337241559/2/100", "0.574049"),
        ],
    ] * 2
    assert [title.find("adjust-transform") for title in titles] == [
        None,
        None,
        None,
        None,
    ]


def test_cli_accepts_fcpxml_bundle(tmp_path):
    import pycut.cli as cli

    bundle = tmp_path / "Library.fcpxmld"
    bundle.mkdir()
    (bundle / "Info.fcpxml").write_text(
        """<fcpxml version="1.11"><resources>
<format id="r1" frameDuration="1/25s"/>
</resources><library><event><project name="Bundle Project">
<sequence format="r1" duration="50/25s"><spine>
<asset-clip ref="r2" offset="0/25s" start="0/25s" duration="50/25s"/>
</spine></sequence></project></event></library></fcpxml>""",
        encoding="utf-8",
    )
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps({"segments": [{"start": 0, "end": 1, "text": "keep"}]}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"

    result = cli.main(
        [
            str(bundle),
            "--transcript",
            str(transcript),
            "--margin-left",
            "0",
            "--margin-right",
            "0",
            "-o",
            str(output_dir),
        ]
    )

    output = output_dir / "Library.fcpxml"
    assert (result["fcpxml"], ET.parse(output).find(".//project").attrib["name"]) == (
        str(output),
        "Bundle Project",
    )


def test_cli_can_keep_empty_transcript_ranges_in_fcpxml(tmp_path):
    import pycut.cli as cli

    source = tmp_path / "project.fcpxml"
    source.write_text(
        """<fcpxml version="1.11"><resources><format id="r1" frameDuration="1/25s"/></resources>
<library><event><project><sequence format="r1" duration="50/25s"><spine>
<asset-clip ref="r2" offset="0/25s" start="0/25s" duration="50/25s"/>
</spine></sequence></project></event></library></fcpxml>""",
        encoding="utf-8",
    )
    transcript = tmp_path / "transcript.json"
    transcript.write_text(
        json.dumps({"segments": [{"start": 0, "end": 1, "text": ""}]}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"

    cli.main(
        [
            str(source),
            "--transcript",
            str(transcript),
            "--format",
            "fcpxml",
            "--no-filter-empty-segments",
            "--margin-left",
            "0",
            "--margin-right",
            "0",
            "-o",
            str(output_dir),
        ]
    )

    clip = ET.parse(output_dir / "project.fcpxml").find(".//spine/asset-clip")
    assert (clip.attrib["start"], clip.attrib["duration"]) == ("0/25s", "25/25s")
