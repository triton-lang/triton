from triton.experimental.gluon import language as ttgl


def test_shared_linear_layout_shape():
    assert ttgl.SharedLinearLayout([[0, 1], [0, 2]]).shape == [1, 4]
    assert ttgl.SharedLinearLayout([[3, 1], [1, 0], [0, 1]]).shape == [4, 2]
    assert ttgl.SharedLinearLayout([[4]]).shape == [8]
