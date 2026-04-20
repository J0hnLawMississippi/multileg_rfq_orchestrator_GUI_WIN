import multileg_rfq_orchestrator_GUI_WIN as gui


def test_demo_screenshot_flag_parsing() -> None:
    args = gui.parse_cli_args(["--demo-screenshot"])
    assert args.demo_screenshot is True

    args_default = gui.parse_cli_args([])
    assert args_default.demo_screenshot is False


def test_demo_default_legs_and_loader_seed() -> None:
    assert gui.DEMO_SCREENSHOT_DEFAULT_LEGS == [
        "L 1.0 29MAY26-80000-C",
        "S 1.0 26JUN26-82000-C",
    ]

    parsed = gui.parse_legs(gui.DEMO_SCREENSHOT_DEFAULT_LEGS)
    assert len(parsed) == 2
    assert parsed[0].maturity == "29MAY26"
    assert parsed[1].maturity == "26JUN26"

    loader = gui.CoincallInstrumentLoader()
    gui._seed_loader_for_demo(loader)

    assert loader.expiry_labels == ["29MAY26", "26JUN26"]
    assert loader.strikes_by_expiry == {
        "29MAY26": ["80000"],
        "26JUN26": ["82000"],
    }
