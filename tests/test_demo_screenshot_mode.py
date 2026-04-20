import multileg_rfq_orchestrator_GUI_WIN as gui


def test_demo_screenshot_flag_parsing() -> None:
    args = gui.parse_cli_args(["--demo-screenshot"])
    assert args.demo_screenshot is True

    args_with_output = gui.parse_cli_args(
        [
            "--demo-screenshot",
            "--screenshot-path",
            "artifacts/gui.png",
            "--auto-exit-after-ready",
        ]
    )
    assert args_with_output.screenshot_path == "artifacts/gui.png"
    assert args_with_output.auto_exit_after_ready is True

    args_default = gui.parse_cli_args([])
    assert args_default.demo_screenshot is False
    assert args_default.screenshot_path == ""
    assert args_default.auto_exit_after_ready is False


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


def test_build_mock_price_result_shape() -> None:
    result = gui._build_mock_price_result(
        leg_specs=gui.DEMO_SCREENSHOT_DEFAULT_LEGS,
        spot=81234.0,
        total_usd=425.0,
        vol_shift=0.0,
    )
    assert result.total_usd == 425.0
    assert result.spot == 81234.0
    assert len(result.legs) == 2
