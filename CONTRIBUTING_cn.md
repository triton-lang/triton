[English](./CONTRIBUTING.md)

# FlagTree 贡献者指南

感谢您对 FlagTree 的兴趣！我们使用 GitHub 来托管代码、管理问题和处理拉取请求。在贡献之前，请阅读以下指南。

## 错误报告

请使用 GitHub 的 Issues 来报告错误。在报告错误时，请提供：
- 简单摘要
- 复现步骤
- 确保描述具体且准确
- 如果可以提供一些示例代码将会很有帮助

## 代码贡献

在提交拉取请求时，贡献者应描述所做的更改以及更改的原因。如果可以设计测试用例，请提供相应测试。拉取请求在合并前需要 __一位__ 成员的批准，而且需要通过代码的持续集成检查。

### 代码格式检查

代码格式检查使用 pre-commit。

```shell
python3 -m pip install pre-commit
cd ${YOUR_CODE_DIR}/flagtree
pre-commit install
pre-commit
```

### 单元测试

安装完成后可以在后端目录下运行单元测试：
```shell
cd third_party/backendxxx/python/test/unit
python3 -m pytest -s
```

### 后端接入

请联系核心开发团队。

## 证书

FlagTree 使用 [MIT license](/LICENSE)。
