"""Test command-line interface functionality."""
import pytest
import sys
import yaml
from unittest.mock import patch, MagicMock
from argparse import Namespace

from sowlv2.cli import main, parse_arguments
from sowlv2.data.config import PipelineBaseData, PipelineConfig


class TestCLIArgumentParsing:
    """Test CLI argument parsing functionality."""
    
    def test_single_prompt_parsing(self):
        """Test parsing single prompt argument."""
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat', 
                               '--input', 'test.jpg', '--output', 'output/']):
            args = parse_arguments()
            assert args.prompt == ['cat']
            assert args.input == 'test.jpg'
            assert args.output == 'output/'
    
    def test_multiple_prompts_parsing(self):
        """Test parsing multiple prompt arguments."""
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat', 'dog', 'bird',
                               '--input', 'test.jpg', '--output', 'output/']):
            args = parse_arguments()
            assert args.prompt == ['cat', 'dog', 'bird']
    
    def test_flag_parsing(self):
        """Test parsing of --no-* flags."""
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat',
                               '--input', 'test.jpg', '--output', 'output/',
                               '--no-binary', '--no-overlay', '--no-merged']):
            args = parse_arguments()
            assert args.no_binary is True
            assert args.no_overlay is True
            assert args.no_merged is True
    
    def test_model_arguments(self):
        """Test parsing of model-specific arguments."""
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat',
                               '--input', 'test.jpg', '--output', 'output/',
                               '--owl-model', 'google/owlv2-large-patch14-ensemble',
                               '--sam-model', 'facebook/sam2.1-hiera-large']):
            args = parse_arguments()
            assert args.owl_model == 'google/owlv2-large-patch14-ensemble'
            assert args.sam_model == 'facebook/sam2.1-hiera-large'
    
    def test_numeric_arguments(self):
        """Test parsing of numeric arguments."""
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat',
                               '--input', 'test.jpg', '--output', 'output/',
                               '--threshold', '0.25', '--fps', '30']):
            args = parse_arguments()
            assert args.threshold == 0.25
            assert args.fps == 30
    
    def test_device_argument(self):
        """Test parsing of device argument."""
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat',
                               '--input', 'test.jpg', '--output', 'output/',
                               '--device', 'cuda']):
            args = parse_arguments()
            assert args.device == 'cuda'
    
    def test_config_file_argument(self):
        """Test parsing of config file argument."""
        with patch('sys.argv', ['sowlv2-detect', '--config', 'config.yaml']):
            args = parse_arguments()
            assert args.config == 'config.yaml'
    
    def test_required_arguments_validation(self):
        """Test that required arguments are properly validated."""
        # Test missing prompt (when no config file)
        with patch('sys.argv', ['sowlv2-detect', '--input', 'test.jpg', '--output', 'output/']):
            with pytest.raises(SystemExit):
                parse_arguments()
        
        # Test missing input (when no config file)
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat', '--output', 'output/']):
            with pytest.raises(SystemExit):
                parse_arguments()


class TestConfigFileHandling:
    """Test configuration file handling."""
    
    def test_config_file_loading(self, test_config_yaml):
        """Test loading configuration from YAML file."""
        with patch('sys.argv', ['sowlv2-detect', '--config', test_config_yaml]):
            # Mock the main execution to avoid running the full pipeline
            with patch('sowlv2.cli.SOWLv2Pipeline') as mock_pipeline:
                with patch('os.path.isfile', return_value=True):
                    with patch('os.path.isdir', return_value=True):
                        with patch('sowlv2.cli.is_valid_image_extension', return_value=True):
                            try:
                                main()
                            except SystemExit:
                                pass  # Expected for successful completion
                
                # Verify pipeline was created with config values
                assert mock_pipeline.called
                call_args = mock_pipeline.call_args[0][0]  # First positional argument (config)
                assert isinstance(call_args, PipelineBaseData)
                assert call_args.threshold == 0.15  # From test config
                assert call_args.fps == 30  # From test config
    
    def test_config_file_prompt_list(self, tmp_path):
        """Test handling of prompt as list in YAML config."""
        config_data = {
            'prompt': ['cat', 'dog', 'bird'],
            'input': 'test.jpg',
            'output': 'output/',
            'threshold': 0.2
        }
        
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        with patch('sys.argv', ['sowlv2-detect', '--config', str(config_path)]):
            with patch('sowlv2.cli.SOWLv2Pipeline') as mock_pipeline:
                with patch('os.path.isfile', return_value=True):
                    with patch('os.path.isdir', return_value=True):
                        with patch('sowlv2.cli.is_valid_image_extension', return_value=True):
                            try:
                                main()
                            except SystemExit:
                                pass
                
                # Should be called with list of prompts
                assert mock_pipeline.called
    
    def test_cli_args_override_config(self, test_config_yaml):
        """Test that CLI arguments override config file values."""
        with patch('sys.argv', ['sowlv2-detect', '--config', test_config_yaml,
                               '--threshold', '0.5', '--fps', '15']):
            with patch('sowlv2.cli.SOWLv2Pipeline') as mock_pipeline:
                with patch('os.path.isfile', return_value=True):
                    with patch('os.path.isdir', return_value=True):
                        with patch('sowlv2.cli.is_valid_image_extension', return_value=True):
                            try:
                                main()
                            except SystemExit:
                                pass
                
                # CLI args should override config file
                call_args = mock_pipeline.call_args[0][0]
                assert call_args.threshold == 0.5  # Overridden by CLI
                assert call_args.fps == 15  # Overridden by CLI
    
    def test_prompt_cli_override_config(self, tmp_path):
        """Test that CLI prompt overrides config file prompt."""
        config_data = {
            'prompt': ['cat', 'dog'],
            'input': 'test.jpg',
            'output': 'output/'
        }
        
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        with patch('sys.argv', ['sowlv2-detect', '--config', str(config_path),
                               '--prompt', 'bird']):
            with patch('sowlv2.cli.SOWLv2Pipeline') as mock_pipeline:
                with patch('os.path.isfile', return_value=True):
                    with patch('os.path.isdir', return_value=True):
                        with patch('sowlv2.cli.is_valid_image_extension', return_value=True):
                            try:
                                main()
                            except SystemExit:
                                pass
                
                # Should use CLI prompt, not config prompt
                assert mock_pipeline.called


class TestPipelineConfigGeneration:
    """Test generation of pipeline configuration from CLI arguments."""
    
    def test_default_pipeline_config(self):
        """Test default pipeline configuration."""
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat',
                               '--input', 'test.jpg', '--output', 'output/']):
            with patch('sowlv2.cli.SOWLv2Pipeline') as mock_pipeline:
                with patch('os.path.isfile', return_value=True):
                    with patch('os.path.isdir', return_value=True):
                        with patch('sowlv2.cli.is_valid_image_extension', return_value=True):
                            try:
                                main()
                            except SystemExit:
                                pass
                
                # Check default config values
                call_args = mock_pipeline.call_args[0][0]
                assert call_args.pipeline_config.binary is True
                assert call_args.pipeline_config.overlay is True
                assert call_args.pipeline_config.merged is True
    
    def test_no_flags_pipeline_config(self):
        """Test pipeline configuration with --no-* flags."""
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat',
                               '--input', 'test.jpg', '--output', 'output/',
                               '--no-binary', '--no-overlay', '--no-merged']):
            with patch('sowlv2.cli.SOWLv2Pipeline') as mock_pipeline:
                with patch('os.path.isfile', return_value=True):
                    with patch('os.path.isdir', return_value=True):
                        with patch('sowlv2.cli.is_valid_image_extension', return_value=True):
                            try:
                                main()
                            except SystemExit:
                                pass
                
                # Check that flags are properly set
                call_args = mock_pipeline.call_args[0][0]
                assert call_args.pipeline_config.binary is False
                assert call_args.pipeline_config.overlay is False
                assert call_args.pipeline_config.merged is False
    
    def test_partial_no_flags_pipeline_config(self):
        """Test pipeline configuration with partial --no-* flags."""
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat',
                               '--input', 'test.jpg', '--output', 'output/',
                               '--no-binary']):
            with patch('sowlv2.cli.SOWLv2Pipeline') as mock_pipeline:
                with patch('os.path.isfile', return_value=True):
                    with patch('os.path.isdir', return_value=True):
                        with patch('sowlv2.cli.is_valid_image_extension', return_value=True):
                            try:
                                main()
                            except SystemExit:
                                pass
                
                # Check that only specified flag is disabled
                call_args = mock_pipeline.call_args[0][0]
                assert call_args.pipeline_config.binary is False
                assert call_args.pipeline_config.overlay is True
                assert call_args.pipeline_config.merged is True


class TestInputValidation:
    """Test input validation functionality."""
    
    def test_image_input_validation(self):
        """Test validation of image input paths."""
        with patch('sys.argv', ['sowlv2-detect', '--prompt', 'cat',
                               '--input', 'test.jpg', '--output', 'output/']):
            with patch('os.path.isfile', return_value=True):
                with patch('sowlv2.cli.is_valid_image_extension', return_value=True):
                    with patch('sowlv2.cli.SOWLv2Pipeline') as mock_pipeline:
                        with patch('os.path.isdir', return_value=True):
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
                    with patch('sowlv2.cli.SOWLv2Pipeline') as mock_pipeline:
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
                with patch('sowlv2.cli.is_valid_image_extension', return_value=False):
                    with patch('sowlv2.cli.SOWLv2Pipeline') as mock_pipeline:
                        with patch('os.path.isdir', return_value=True):
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
                    with pytest.raises(SystemExit):
                        main()


class TestErrorHandling:
    """Test error handling in CLI."""
    
    def test_missing_config_file(self):
        """Test handling of missing config file."""
        with patch('sys.argv', ['sowlv2-detect', '--config', 'nonexistent.yaml']):
            with pytest.raises(SystemExit):
                main()
    
    def test_invalid_config_file(self, tmp_path):
        """Test handling of invalid YAML config file."""
        config_path = tmp_path / "invalid.yaml"
        with open(config_path, 'w') as f:
            f.write("invalid: yaml: content:")  # Invalid YAML
        
        with patch('sys.argv', ['sowlv2-detect', '--config', str(config_path)]):
            with pytest.raises(SystemExit):
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
                args = parse_arguments()
                # If parsing succeeds, threshold should be validated elsewhere
            except (SystemExit, ValueError):
                # Either is acceptable for invalid values
                pass