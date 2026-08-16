import ast
import pathlib
import unittest


WAN21_ROOT = pathlib.Path(__file__).resolve().parents[1]
WRAPPER_PATH = WAN21_ROOT / "wan_utils" / "wan_wrapper.py"


class WanWrapperConfigTest(unittest.TestCase):
    def test_tri_region_rope_flag_reaches_causal_model_call(self):
        module = ast.parse(WRAPPER_PATH.read_text())
        wrapper_class = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "WanDiffusionWrapper"
        )
        initializer = next(
            node
            for node in wrapper_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        argument_names = [argument.arg for argument in initializer.args.args]
        self.assertIn("tri_region_rope_enabled", argument_names)

        from_pretrained = next(
            node
            for node in ast.walk(initializer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_pretrained"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "CausalWanModel"
        )
        keyword = next(
            item
            for item in from_pretrained.keywords
            if item.arg == "tri_region_rope_enabled"
        )
        self.assertIsInstance(keyword.value, ast.Name)
        self.assertEqual(keyword.value.id, "tri_region_rope_enabled")


if __name__ == "__main__":
    unittest.main()
