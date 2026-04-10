# Plan: Triton에서 GEMM + Reduce-Scatter Overlap 가능성 검증 및 IR 분석

## Goal

다음 작업을 수행한다.

1. **GEMM + Reduce-Scatter fused kernel**을 **Triton language**로 작성한다.  
   - 필수 제약: **torch.distributed의 symmetric memory feature를 반드시 사용**
2. **Triton compiler의 software pipeline 관련 pass**를 활용하여  
   **GEMM과 Reduce-Scatter의 overlapping** 최적화가 가능한지 시도한다.
3. **overlapping 적용 전/후의 IR 차이**를 출력한다.
4. 만약 현재 Triton의 software pipeline pass만으로 overlapping이 불가능하면,
   - 불가능한 이유를 구조적으로 분석하고
   - 이를 해결하기 위해 어떤 compiler pass / IR transformation / scheduling pass가 필요한지 정리한
   **보고서(markdown)** 를 저장한다.
5. 반대로 overlapping이 가능하면,
   - 성공한 IR 예제
   - 적용 방법
   - 적용 전/후 차이
   - 성능상 기대 효과 및 한계
   를 포함한 **보고서(markdown)** 를 저장한다.

---

## Documentation Requirement

이번 작업에서 **조사하고 확인한 사실은 모두 markdown 파일로 남긴다.**  
최종 보고서뿐 아니라, 중간 조사 결과, 실험 결과, 실패 원인, 가설과 검증 결과도 전부 markdown으로 저장한다.

### Mandatory rule
- 확인된 사실, 관찰 결과, 소스 코드 위치, pass 동작, IR 특징, runtime 제약, 실패 원인, 설계 제안은 **반드시 markdown 파일에 기록**할 것
- 구두 요약이나 임시 메모로 끝내지 말 것
- 각 에이전트는 자기 작업 결과를 **자기 전용 markdown 보고서**에 누적 기록할 것
- 최종 에이전트는 이 markdown 파일들을 읽어 종합 보고서를 작성할 것

### Distinction rule
각 markdown 파일에는 아래 구분을 명확히 둘 것.
- **Confirmed facts**
- **Observed behavior**
- **Hypotheses**
- **Open questions**
- **Next actions**

확인되지 않은 추정은 fact처럼 쓰지 말 것.

---

## Output Directory Structure

모든 조사 결과는 아래 구조로 정리한다.

```text
artifacts/
  ir_before/
  ir_after/
  pass_logs/
notes/
  agent_findings/
    compiler_pass_inspection.md
    symmetric_memory_investigation.md
    reduce_scatter_runtime_model.md
    kernel_design.md
    ir_before_summary.md
    ir_comparison.md
    failure_analysis_and_pass_design.md
    execution_log.md
report_gemm_reduce_scatter_overlap.md
diff_ir.txt
gemm_reduce_scatter_triton.py
