#include <gtest/gtest.h>
#include <Python.h>

TEST(MetalDriver, GetDeviceProperties) {
  Py_Initialize();
  PyObject *pName = PyUnicode_FromString("third_party.metal.backend.driver");
  PyObject *pModule = PyImport_Import(pName);
  Py_DECREF(pName);

  ASSERT_NE(pModule, nullptr);

  PyObject *pFunc = PyObject_GetAttrString(pModule, "get_device_properties");
  ASSERT_NE(pFunc, nullptr);

  PyObject *pArgs = PyTuple_New(1);
  PyTuple_SetItem(pArgs, 0, PyLong_FromLong(0));

  PyObject *pValue = PyObject_CallObject(pFunc, pArgs);
  Py_DECREF(pArgs);

  ASSERT_NE(pValue, nullptr);

  PyObject *pNameObj = PyDict_GetItemString(pValue, "name");
  ASSERT_NE(pNameObj, nullptr);
  
  const char *name = PyUnicode_AsUTF8(pNameObj);
  EXPECT_NE(name, nullptr);
  EXPECT_GT(strlen(name), 0);

  Py_DECREF(pValue);
  Py_DECREF(pFunc);
  Py_DECREF(pModule);

  Py_Finalize();
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
