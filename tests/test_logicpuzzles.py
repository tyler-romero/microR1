import unittest

import logicpuzzles


class TaskPresetTest(unittest.TestCase):
    def test_default_messages_put_format_instruction_in_system_prompt(self):
        messages = logicpuzzles.make_messages('Solve this.')

        self.assertEqual(messages[0], {'role': 'system', 'content': logicpuzzles.system_prompt})
        self.assertEqual(messages[1], {'role': 'user', 'content': 'Solve this.'})

    def test_qwen_messages_keep_identity_in_system_prompt(self):
        messages = logicpuzzles.make_messages('Solve this.', message_style='qwen')

        self.assertEqual(messages[0], {'role': 'system', 'content': logicpuzzles.system_prompt_qwen})
        self.assertEqual(messages[1]['role'], 'user')
        self.assertEqual(messages[1]['content'], f'Solve this.\n{logicpuzzles.system_prompt}')

    def test_qwen_4090_preset_contains_the_tuned_curriculum(self):
        preset = logicpuzzles.task_presets['qwen_4090']

        self.assertEqual(preset.message_style, 'qwen')
        self.assertEqual(preset.linear_max_coefficient, 25)
        self.assertEqual(preset.nim_max_pile_size, 25)
        self.assertEqual(preset.josephus_max_k, 5)
        self.assertTrue(preset.verbose_prompts)

    def test_dataset_applies_selected_message_style(self):
        default_messages, _ = next(logicpuzzles.gen_dataset('linear_equations'))
        qwen_messages, _ = next(logicpuzzles.gen_dataset('linear_equations', task_preset='qwen_4090'))

        self.assertEqual(default_messages[0]['content'], logicpuzzles.system_prompt)
        self.assertEqual(qwen_messages[0]['content'], logicpuzzles.system_prompt_qwen)
        self.assertIn(logicpuzzles.system_prompt, qwen_messages[1]['content'])

    def test_unknown_presets_and_message_styles_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown task preset'):
            next(logicpuzzles.gen_dataset('all', task_preset='missing'))
        with self.assertRaisesRegex(ValueError, 'Unknown message style'):
            logicpuzzles.make_messages('Solve this.', message_style='missing')


if __name__ == '__main__':
    unittest.main()
