# Bug Fix Plan: `get_cache_manager` Crashes on Non-Hex Keys

**Issue:** [triton-lang/triton#5013](https://github.com/triton-lang/triton/issues/5013)
**Component:** `python/triton/runtime/cache.py`
**Severity:** Runtime crash (`ValueError`) for any external caller passing non-hex keys

---

## Table of Contents

- [Summary](#summary)
- [Root Cause Analysis](#root-cause-analysis)
- [Impact Assessment](#impact-assessment)
- [Fix Alternatives](#fix-alternatives)
  - [Option A: Graceful Fallback in `_base32` (Recommended)](#option-a-graceful-fallback-in-_base32-recommended)
  - [Option B: Always Hash Before Encoding](#option-b-always-hash-before-encoding)
  - [Option C: Encode as UTF-8 Instead of Hex-Decoding](#option-c-encode-as-utf-8-instead-of-hex-decoding)
- [Chosen Approach](#chosen-approach)
- [Implementation Details](#implementation-details)
- [Testing Strategy](#testing-strategy)
- [Files Changed](#files-changed)

---

## Summary

`triton.runtime.cache.get_cache_manager(key)` raises a `ValueError` when called
with a non-hexadecimal string key. The error originates in the private `_base32()`
helper which unconditionally calls `bytes.fromhex(key)`. This breaks third-party
libraries (e.g., [LigerKernel](https://github.com/linkedin/Liger-Kernel)) that
use `get_cache_manager` as a public API with arbitrary string keys.

**Reproducer** (from the issue):

```python
from triton.runtime.cache import get_cache_manager
cache_manager = get_cache_manager(key="test_hash")
# ValueError: non-hexadecimal number found in fromhex() arg at position 0
```

---

## Root Cause Analysis

### History

| PR | Date | Change | Effect |
|----|------|--------|--------|
| Pre-[#4553](https://github.com/triton-lang/triton/pull/4553) | Before Aug 2024 | `get_cache_manager(key)` passed `key` directly to `FileCacheManager` | Any string worked |
| [#4553](https://github.com/triton-lang/triton/pull/4553) | Aug 2024 | Introduced `_base64(key)` using `bytes.fromhex(key)` for shorter directory names | **Broke non-hex keys** |
| [#5088](https://github.com/triton-lang/triton/pull/5088) | Nov 2024 | Renamed `_base64` to `_base32` (avoid leading hyphens) | Bug preserved |

### The Buggy Code Path

```
get_cache_manager(key)       # cache.py:254 - public API, any string
  -> _base32(key)            # cache.py:249 - assumes hex string
    -> bytes.fromhex(key)    # CRASH if key is not hex
```

The comment on line 250 explicitly acknowledges the assumption:

```python
def _base32(key):
    # Assume key is a hex string.
    return base64.b32encode(bytes.fromhex(key)).decode("utf-8").rstrip("=")
```

### Why Internal Callers Are Unaffected

All internal call sites hash their keys with `hashlib.sha256(...).hexdigest()`
before passing to `get_cache_manager`, producing valid hex strings:

| Caller | Location | Key Source |
|--------|----------|------------|
| `compiler.py:251` | `get_cache_manager(hash)` | `hashlib.sha256(...).hexdigest()` |
| `autotuner.py:191` | `get_cache_manager(cache_key)` | `hashlib.sha256(...).hexdigest()` |
| `build.py:78` | `get_cache_manager(key)` | `hashlib.sha256(...).hexdigest()` |
| `compiler.py:257` | `get_override_manager(src.hash())` | `hashlib.sha256(...).hexdigest()` |

The bug only manifests for **external callers** who use `get_cache_manager` as a
public API without pre-hashing their keys.

---

## Impact Assessment

- **Who is affected:** External/third-party code calling `get_cache_manager()`
  with arbitrary string keys (e.g., LigerKernel `apply_liger_triton_cache_manager`).
- **Internal code:** Unaffected (all internal callers produce hex keys).
- **Cache invalidation:** Fix must NOT invalidate existing caches for hex keys
  (backward compatibility).

---

## Fix Alternatives

### Option A: Graceful Fallback in `_base32` (Recommended)

Catch `ValueError` from `bytes.fromhex()` and fall back to hashing the key:

```python
def _base32(key):
    try:
        key_bytes = bytes.fromhex(key)
    except (ValueError, AttributeError):
        key_bytes = hashlib.sha256(key.encode("utf-8")).digest()
    return base64.b32encode(key_bytes).decode("utf-8").rstrip("=")
```

| Pros | Cons |
|------|------|
| **Backward compatible** - hex keys produce identical output | Two code paths (minor) |
| Minimal change (3 lines) | |
| No cache invalidation for existing users | |
| Handles any string input gracefully | |
| Follows defensive programming pattern | |

### Option B: Always Hash Before Encoding

Always run `hashlib.sha256` on the key before base32 encoding:

```python
def _base32(key):
    key_bytes = hashlib.sha256(key.encode("utf-8")).digest()
    return base64.b32encode(key_bytes).decode("utf-8").rstrip("=")
```

| Pros | Cons |
|------|------|
| Single code path, simple | **Breaks backward compatibility** - all cache dirs change |
| Consistent output length | **Invalidates all existing caches** for every user |
| Always works for any input | Double-hashing for internal callers (already hex) |

### Option C: Encode as UTF-8 Instead of Hex-Decoding

Use the raw string bytes instead of hex-decoding:

```python
def _base32(key):
    return base64.b32encode(key.encode("utf-8")).decode("utf-8").rstrip("=")
```

| Pros | Cons |
|------|------|
| Simplest change (1 line) | **Breaks backward compatibility** - all cache dirs change |
| Always works | Longer output (defeats original purpose of PR #4553) |
| | **Invalidates all existing caches** |

---

## Chosen Approach

**Option A** is the recommended fix because it:

1. **Preserves backward compatibility** - existing hex keys (from all internal
   callers) produce identical base32 output, so no cache invalidation occurs.
2. **Fixes the reported bug** - non-hex keys are gracefully handled by hashing
   to fixed-length bytes via SHA-256.
3. **Minimal risk** - the try/except only activates for non-hex input; the
   existing fast path is unchanged.
4. **Follows repository conventions** - defensive, minimal fix with no unnecessary
   side effects (per `CONTRIBUTING.md` guidelines on functional bug fixes).

---

## Implementation Details

### Change 1: Fix `_base32` in `cache.py`

**File:** `python/triton/runtime/cache.py`, lines 249-251

**Before:**
```python
def _base32(key):
    # Assume key is a hex string.
    return base64.b32encode(bytes.fromhex(key)).decode("utf-8").rstrip("=")
```

**After:**
```python
def _base32(key):
    # Try to decode as hex (all internal callers use hexdigest keys).
    # Fall back to hashing for arbitrary string keys from external callers.
    try:
        key_bytes = bytes.fromhex(key)
    except (ValueError, AttributeError):
        key_bytes = hashlib.sha256(key.encode("utf-8")).digest()
    return base64.b32encode(key_bytes).decode("utf-8").rstrip("=")
```

### Change 2: Add Unit Test

**File:** `python/test/unit/runtime/test_cache.py`

Add a test that verifies `get_cache_manager` works with both hex and non-hex keys,
confirming the fix for the reported issue.

---

## Testing Strategy

1. **Unit test** - Verify `_base32` handles hex keys, non-hex keys, and edge cases.
2. **Backward compatibility** - Verify hex keys produce the same base32 output
   as before (no cache invalidation).
3. **Pre-commit checks** - Run `pre-commit run --from-ref origin/main --to-ref HEAD`.

---

## Files Changed

| File | Change |
|------|--------|
| `python/triton/runtime/cache.py` | Fix `_base32` to handle non-hex keys |
| `python/test/unit/runtime/test_cache.py` | Add regression test |
| `bugfix-plan-issue-5013.md` | This plan document |
