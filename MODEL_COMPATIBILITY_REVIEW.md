# Model Compatibility Review Notes

This note records the review issues found after adding CIFAR-10 `small_cnn`
and Tiny-ImageNet `resnet34` support.

## 1. `cleansers_tool_box/sentinet.py` still assumes ResNet GradCAM

Status: known issue, not modified for now.

The defense path in `other_defenses_tool_box/sentinet.py` now detects
`small_cnn` and uses GradCAM on `block3` through the `last` alias.

However, the cleanser path in `cleansers_tool_box/sentinet.py` still hardcodes
GradCAM as a ResNet:

```python
GradCAM(dict(type='resnet', arch=self.model.module, layer_name='layer4', ...))
```

This is safe for ResNet models, including `resnet34`, but it can fail for
`SmallCNN_cifar10` because SmallCNN has `block1`, `block2`, and `block3`, not
`layer4`.

Impact:

- `other_defense.py -defense=SentiNet -model=small_cnn` should use the updated
  SentiNet path.
- `cleanser.py -cleanser=SentiNet -model=small_cnn` may fail during GradCAM
  target-layer lookup.
- Existing ResNet paths are not affected.

Suggested future fix:

- Share a small GradCAM target-layer helper between the defense and cleanser
  SentiNet implementations, or duplicate the SmallCNN branch in
  `cleansers_tool_box/sentinet.py`.

## 2. Some CLI `-model choices` allow unsupported dataset/model combinations

Status: known issue, not modified for now.

Several entrypoints accept `small_cnn` in their argparse `choices`, even when
the current dataset is not CIFAR-10. The actual restriction is enforced later
in `utils/supervisor.py`:

```python
small_cnn is only supported for dataset='cifar10'
```

Impact:

- This does not silently run the wrong model.
- Invalid combinations fail when `supervisor.get_arch(args)` is called.
- Batch jobs may fail later than expected because argparse accepts the model
  name before dataset/model compatibility is checked.

Suggested future fix:

- Add a centralized `validate_model_dataset(args)` helper and call it after
  argument parsing in the main entrypoints.
- Keep `small_cnn` limited to CIFAR-10 unless a new dataset-specific SmallCNN
  variant is added.

## 3. `utils/resnet.py` had mixed line endings

Status: fixed separately from functional model changes.

The file had both CRLF and LF line endings after the ResNet34 Tiny-ImageNet
factory was added. This does not affect Python execution, but it makes diffs
noisier and reviews harder.

Impact:

- No runtime behavior change.
- Git diffs can show lines as modified even when only line endings changed.

Fix applied:

- Normalize `utils/resnet.py` line endings consistently to LF.
