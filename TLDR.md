## TLDR: Which workflow (CLI/MCP) best helps to monitor and keep the conversation happening in our open source communities safe and effective for dev? 

This evaluation compares two ways of building a GitHub community-health triage agent: a CLI workflow based on `gh` and an MCP workflow based on the GitHub MCP Server.

 The agent retrieves a discussion thread, detects toxic or discouraging content, assigns a label and severity, drafts a maintainer response, and posts a report.

## Goal
The core question is which workflow produces better results at lower cost and with fewer failures. 

The evaluation measures retrieval completeness, response quality, latency, token usage, and whether the agent stays within the sandbox boundary. In this project, final reports are posted as issues in the evaluation repository, not in live OSS repositories: https://github.com/imhurl23/community_monitor_eval/issues

## Dataset
The evaluated runs use samples from the pandas-dev/pandas repository, although the /curate_dataset component can build datasets for other open-source projects. Issues and pull requests are sorted into strata such as clearly toxic, borderline, heated-but-not-toxic, and control using a toxicity classifier tuned for open-source discussion. Then the dataset is compiled ensuring representation from all strata levels. 

A human review step then checks the model recommendation and records ground-truth toxicity, toxicity type, and a maintainer response when one is needed. Each row includes thread metadata, a binary toxicity label, and, for toxic examples, a severity level, a problematic snippet, and a gold-standard maintainer reply.

## Scoring
The scorer set checks for scope containment, report posting, snippet grounding, retrieval completeness, label accuracy, and de-escalation quality. This combination helps separate bad retrieval from bad writing, which is essential for understanding why a workflow succeeds or fails.

## Why It Matters
Open-source toxicity is often subtle and contextual, especially in forms like entitlement, passive aggression, and gatekeeping. A strong community-health agent has to retrieve enough context to avoid false negatives, but it also has to avoid over-flagging ordinary disagreement and violating contraints on where it should operate. 
