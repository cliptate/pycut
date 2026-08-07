import json
import xml.etree.ElementTree as ET


def test_cli_rough_cuts_fcpxml_from_transcript(tmp_path):
    import pycut.cli as cli

    source = tmp_path / "project.fcpxml"
    source.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<fcpxml version="1.11">
  <resources>
    <format id="r1" frameDuration="1/25s"/>
    <asset id="r2" name="Interview" start="0/25s" duration="250/25s" format="r1">
      <media-rep kind="original-media" src="file:///tmp/interview.mov"/>
    </asset>
  </resources>
  <library><event name="Event"><project name="Rough Cut">
    <sequence format="r1" duration="250/25s"><spine>
      <asset-clip ref="r2" name="Interview" offset="0/25s" duration="250/25s"/>
    </spine></sequence>
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
    assert (
        result["fcpxml"],
        root.find(".//project").attrib["name"],
        [(clip.attrib["offset"], clip.attrib["start"], clip.attrib["duration"]) for clip in clips],
    ) == (
        str(output),
        "Rough Cut",
        [
            ("0/25s", "25/25s", "25/25s"),
            ("25/25s", "100/25s", "25/25s"),
        ],
    )


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
