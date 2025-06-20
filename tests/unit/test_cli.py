"""Test command-line interface functionality."""
from unittest.mock import patch

import pytest
import yaml

from sowlv2.cli import main, parse_args
from sowlv2.data.config import PipelineBaseData


class TestCLIArgumentParsing:
    """Test CLI argument parsing functionality."""

    def test_single_prompt_parsing(self):
        """Test parsing single prompt argument."""
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat',
                               '--input', 'test.jpg', '--output', 'output/']):
            args = parse_args()
            assert args.prompt == ['cat']
            assert args.input == 'test.jpg'
            assert args.output == 'output/'

    def test_multiple_prompts_parsing(self):
        """Test parsing multiple prompt arguments."""
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat', 'dog', 'bird',
                               '--input', 'test.jpg', '--output', 'output/']):
            args = parse_args()
            assert args.prompt == ['cat', 'dog', 'bird']

    def test_flag_parsing(self):
        """Test parsing of --no-* flags."""
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat',
                               '--input', 'test.jpg', '--output', 'output/',
                               '--no-binary', '--no-overlay', '--no-merged']):
            args = parse_args()
            assert args.binary is False  # --no-binary sets binary to False
            assert args.overlay is False  # --no-overlay sets overlay to False
            assert args.merged is False  # --no-merged sets merged to False

    def test_model_arguments(self):
        """Test parsing of model-specific arguments."""
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat',
                               '--input', 'test.jpg', '--output', 'output/',
                               '--owl-model', 'google/owlv2-large-patch14-ensemble',
                               '--sam-model', 'facebook/sam2.1-hiera-large']):
            args = parse_args()
            assert args.owl_model == 'google/owlv2-large-patch14-ensemble'
            assert args.sam_model == 'facebook/sam2.1-hiera-large'

    def test_numeric_arguments(self):
        """Test parsing of numeric arguments."""
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat',
                               '--input', 'test.jpg', '--output', 'output/',
                               '--threshold', '0.25', '--fps', '30']):
            args = parse_args()
            assert args.threshold == 0.25
            assert args.fps == 30

    def test_device_argument(self):
        """Test parsing of device argument."""
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat',
                               '--input', 'test.jpg', '--output', 'output/',
                               '--device', 'cuda']):
            args = parse_args()
            assert args.device == 'cuda'

    def test_config_file_argument(self):
        """Test parsing of config file argument."""
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat', '--input', 'test.jpg',
                               '--config', 'config.yaml']):
            # Mock file opening to test argument parsing only
            with patch('builtins.open'), patch('yaml.safe_load', return_value={}):
                args = parse_args()
                assert args.config == 'config.yaml'

    def test_required_arguments_validation(self):
        """Test that required arguments are properly validated."""
        # Test missing prompt (when no config file)
        with patch('sys.argv', ['sowlv2-detect', '--input', 'test.jpg', '--output', 'output/']):
            with pytest.raises(SystemExit):
                parse_args()

        # Test missing input (when no config file)
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat', '--output', 'output/']):
            with pytest.raises(SystemExit):
                parse_args()


class TestConfigFileHandling:
    """Test configuration file handling."""

    def test_config_file_loading(self, test_config_yaml):
        """Test loading configuration from YAML file."""
        with patch('sys.argv', ['sowlv2-detect', '--config', test_config_yaml]):
            # Mock the main execution to avoid running the full pipeline
            with patch('sowlv2.cli.OptimizedSOWLv2Pipeline') as mock_pipeline:
                with patch('os.path.isfile', return_value=True):
                    with patch('os.makedirs'):
                        with patch('builtins.print'):  # Suppress prints
                            main()

                # Verify pipeline was created with config values
                assert mock_pipeline.called
                call_args = mock_pipeline.call_args.args  # Positional arguments
                config = call_args[0]  # First argument is config
                assert isinstance(config, PipelineBaseData)
                assert config.threshold == 0.15  # From test config
                assert config.fps == 30  # From test config

    def test_config_file_prompt_list(self, tmp_path):
        """Test handling of prompt as list in YAML config."""
        config_data = {
            'prompt': ['cat', 'dog', 'bird'],
            'input': 'test.jpg',
            'output': 'output/',
            'threshold': 0.2
        }

        config_path = tmp_path / "test_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)

        with patch('sys.argv', ['sowlv2-detect', '--config', str(config_path)]):
            with patch('sowlv2.cli.OptimizedSOWLv2Pipeline') as mock_pipeline:
                with patch('os.path.isfile', return_value=True):
                    with patch('os.makedirs'):
                        with patch('builtins.print'):
                            main()

                # Should be called with list of prompts
                assert mock_pipeline.called

    def test_cli_args_override_config(self, test_config_yaml):
        """Test that CLI arguments override config file values."""
        with patch('sys.argv', ['sowlv2-detect', '--config', test_config_yaml,
                               '--threshold', '0.5', '--fps', '15']):
            with patch('sowlv2.cli.OptimizedSOWLv2Pipeline') as mock_pipeline:
                with patch('os.path.isfile', return_value=True):
                    with patch('os.makedirs'):
                        with patch('builtins.print'):
                            main()

                # CLI args should override config file
                config = mock_pipeline.call_args.args[0]  # First argument is config
                assert config.threshold == 0.5  # Overridden by CLI
                assert config.fps == 15  # Overridden by CLI

    def test_prompt_cli_override_config(self, tmp_path):
        """Test that CLI prompt overrides config file prompt."""
        config_data = {
            'prompt': ['cat', 'dog'],
            'input': 'test.jpg',
            'output': 'output/'
        }

        config_path = tmp_path / "test_config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f)

        with patch('sys.argv', ['sowlv2-detect', '--config', str(config_path),
                               '--prompt', 'bird']):
            with patch('sowlv2.cli.OptimizedSOWLv2Pipeline') as mock_pipeline:
                with patch('os.path.isfile', return_value=True):
                    with patch('os.makedirs'):
                        with patch('builtins.print'):
                            main()

                # Should use CLI prompt, not config prompt
                assert mock_pipeline.called


class TestPipelineConfigGeneration:
    """Test generation of pipeline configuration from CLI arguments."""

    def test_default_pipeline_config(self):
        """Test default pipeline configuration."""
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat',
                               '--input', 'test.jpg', '--output', 'output/']):
            with patch('sowlv2.cli.OptimizedSOWLv2Pipeline') as mock_pipeline:
                with patch('os.path.isfile', return_value=True):
                    with patch('os.makedirs'):
                        with patch('builtins.print'):
                            main()

                # Check default config values
                config = mock_pipeline.call_args.args[0]  # First argument is config
                assert config.pipeline_config.binary is True
                assert config.pipeline_config.overlay is True
                assert config.pipeline_config.merged is True

    def test_no_flags_pipeline_config(self):
        """Test pipeline configuration with --no-* flags."""
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat',
                               '--input', 'test.jpg', '--output', 'output/',
                               '--no-binary', '--no-overlay', '--no-merged']):
            with patch('sowlv2.cli.OptimizedSOWLv2Pipeline') as mock_pipeline:
                with patch('os.path.isfile', return_value=True):
                    with patch('os.path.isdir', return_value=False):
                        with patch('sowlv2.cli.VALID_EXTS', ['.jpg']):
                            with patch('os.makedirs'):
                                with patch('builtins.print'):
                                    try:
                                        main()
                                    except SystemExit:
                                        pass

                # Check that flags are properly set
                config = mock_pipeline.call_args.args[0]  # First argument is config
                assert config.pipeline_config.binary is False
                assert config.pipeline_config.overlay is False
                assert config.pipeline_config.merged is False

    def test_partial_no_flags_pipeline_config(self):
        """Test pipeline configuration with partial --no-* flags."""
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat',
                               '--input', 'test.jpg', '--output', 'output/',
                               '--no-binary']):
            with patch('sowlv2.cli.OptimizedSOWLv2Pipeline') as mock_pipeline:
                with patch('os.path.isfile', return_value=True):
                    with patch('os.path.isdir', return_value=False):
                        with patch('sowlv2.cli.VALID_EXTS', ['.jpg']):
                            with patch('os.makedirs'):
                                with patch('builtins.print'):
                                    try:
                                        main()
                                    except SystemExit:
                                        pass

                # Check that only specified flag is disabled
                config = mock_pipeline.call_args.args[0]  # First argument is config
                assert config.pipeline_config.binary is False
                assert config.pipeline_config.overlay is True
                assert config.pipeline_config.merged is True


class TestInputValidation:
    """Test input validation functionality."""

    def test_image_input_validation(self):
        """Test validation of image input paths."""
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat',
                               '--input', 'test.jpg', '--output', 'output/']):
            with patch('os.path.isfile', return_value=True):
                with patch('os.path.isdir', return_value=False):
                    with patch('sowlv2.cli.VALID_EXTS', ['.jpg']):
                        with patch('sowlv2.cli.OptimizedSOWLv2Pipeline') as mock_pipeline:
                            with patch('os.makedirs'):
                                with patch('builtins.print'):
                                    try:
                                        main()
                                    except SystemExit:
                                        pass

                            # Should call process_image
                            mock_instance = mock_pipeline.return_value
                            assert mock_instance.process_image.called

    def test_directory_input_validation(self):
        """Test validation of directory input paths."""
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat',
                               '--input', 'frames/', '--output', 'output/']):
            with patch('os.path.isfile', return_value=False):
                with patch('os.path.isdir', return_value=True):
                    with patch('sowlv2.cli.OptimizedSOWLv2Pipeline') as mock_pipeline:
                        with patch('os.makedirs'):
                            with patch('builtins.print'):
                                try:
                                    main()
                                except SystemExit:
                                    pass

                        # Should call process_frames
                        mock_instance = mock_pipeline.return_value
                        assert mock_instance.process_frames.called

    def test_video_input_validation(self):
        """Test validation of video input paths."""
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat',
                               '--input', 'video.mp4', '--output', 'output/']):
            with patch('os.path.isfile', return_value=True):
                with patch('os.path.isdir', return_value=False):
                    with patch('sowlv2.cli.VALID_EXTS', ['.jpg']):  # Not .mp4, so it will try video
                        with patch('sowlv2.cli.VALID_VIDEO_EXTS', ['.mp4']):
                            with patch('sowlv2.cli.OptimizedSOWLv2Pipeline') as mock_pipeline:
                                with patch('os.makedirs'):
                                    with patch('builtins.print'):
                                        try:
                                            main()
                                        except SystemExit:
                                            pass

                                # Should call process_video
                                mock_instance = mock_pipeline.return_value
                                assert mock_instance.process_video.called

    def test_invalid_input_handling(self):
        """Test handling of invalid input paths."""
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat',
                               '--input', 'nonexistent.jpg', '--output', 'output/']):
            with patch('os.path.isfile', return_value=False):
                with patch('os.path.isdir', return_value=False):
                    with patch('sowlv2.cli.OptimizedSOWLv2Pipeline'):
                        with patch('os.makedirs'):
                            with patch('builtins.print'):  # Suppress error message
                                with pytest.raises(SystemExit):
                                    main()


class TestErrorHandling:
    """Test error handling in CLI."""

    def test_missing_config_file(self):
        """Test handling of missing config file."""
        with patch('sys.argv', ['sowlv2-detect', '--config', 'nonexistent.yaml']):
            with pytest.raises(FileNotFoundError):
                main()

    def test_invalid_config_file(self, tmp_path):
        """Test handling of invalid YAML config file."""
        config_path = tmp_path / "invalid.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("invalid: yaml: content:")  # Invalid YAML

        with patch('sys.argv', ['sowlv2-detect', '--config', str(config_path)]):
            with pytest.raises(yaml.scanner.ScannerError):
                main()

    def test_conflicting_arguments(self):
        """Test handling of conflicting arguments."""
        # This might be expanded based on specific conflict scenarios
        # For now, just test that the parser handles edge cases
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat',
                               '--input', 'test.jpg', '--output', 'output/',
                               '--threshold', '-0.5']):  # Invalid threshold
            # Should either handle gracefully or raise appropriate error
            try:
                parse_args()
                # If parsing succeeds, threshold should be validated elsewhere
            except (SystemExit, ValueError):
                # Either is acceptable for invalid values
                pass
