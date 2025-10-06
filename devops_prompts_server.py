#!/usr/bin/env python3
"""
DevOps Prompts MCP Server

This server provides reusable prompt templates for various DevOps activities including:
- CI/CD pipeline troubleshooting
- Infrastructure monitoring and alerting
- Security analysis and incident response
- Code review and deployment best practices
- System architecture and performance optimization

Usage:
    python devops_prompts_server.py

The server will run on stdio by default, or use --transport sse --port 8001 for SSE.
"""

import anyio
import click
import logging
import mcp.types as types
from mcp.server.lowlevel import Server
from starlette.requests import Request

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("devops-prompts-server")

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

def create_cicd_pipeline_analysis_prompt(
    pipeline_name: str,
    failure_stage: str,
    error_logs: str,
    tech_stack: str = "Docker, Kubernetes, Jenkins"
) -> list[types.PromptMessage]:
    """Create CI/CD pipeline analysis prompt."""
    text = f"""Please analyze this CI/CD pipeline failure and provide detailed troubleshooting steps:

**Pipeline Information:**
- Pipeline Name: {pipeline_name}
- Failed Stage: {failure_stage}
- Technology Stack: {tech_stack}

**Error Logs:**
```
{error_logs}
```

**Analysis Request:**
1. Identify the root cause of the failure
2. Provide step-by-step troubleshooting instructions
3. Suggest preventive measures for future occurrences
4. Recommend monitoring improvements
5. Include relevant best practices for {tech_stack}

Please format your response with clear sections and actionable recommendations."""
    
    return [types.PromptMessage(role="user", content=types.TextContent(type="text", text=text))]

def create_deployment_strategy_review_prompt(
    application_type: str,
    current_strategy: str,
    environment: str = "production",
    traffic_volume: str = "medium"
) -> list[types.PromptMessage]:
    """Create deployment strategy review prompt."""
    text = f"""Review the current deployment strategy and provide optimization recommendations:

**Application Details:**
- Type: {application_type}
- Environment: {environment}
- Traffic Volume: {traffic_volume}
- Current Strategy: {current_strategy}

**Review Areas:**
1. **Strategy Assessment**: Analyze the current deployment approach
2. **Risk Analysis**: Identify potential risks and mitigation strategies
3. **Performance Impact**: Evaluate deployment impact on system performance
4. **Rollback Planning**: Assess rollback procedures and recovery time
5. **Alternative Strategies**: Suggest blue-green, canary, or rolling deployment options
6. **Monitoring Requirements**: Define deployment monitoring and alerting needs

Please provide detailed recommendations with pros/cons for each suggestion."""
    
    return [types.PromptMessage(role="user", content=types.TextContent(type="text", text=text))]

def create_code_review_checklist_prompt(
    language: str,
    change_type: str,
    security_level: str = "standard"
) -> list[types.PromptMessage]:
    """Create code review checklist prompt."""
    text = f"""Create a comprehensive code review checklist for this change:

**Code Change Details:**
- Programming Language: {language}
- Change Type: {change_type}
- Security Level: {security_level}

**Generate checklist covering:**

**1. Code Quality:**
- Code structure and organization
- Naming conventions and readability
- Performance considerations
- Error handling and edge cases

**2. Security Review:**
- Security vulnerabilities specific to {language}
- Input validation and sanitization
- Authentication and authorization
- Data protection and privacy

**3. Testing Requirements:**
- Unit test coverage
- Integration test scenarios
- Performance test considerations
- Security test recommendations

**4. Documentation:**
- Code comments and documentation
- API documentation updates
- README and setup instructions

**5. DevOps Considerations:**
- Deployment impact assessment
- Configuration management
- Monitoring and logging additions

Please provide a detailed, actionable checklist with specific items to verify."""
    
    return [types.PromptMessage(role="user", content=types.TextContent(type="text", text=text))]

def create_infrastructure_incident_response_prompt(
    service_name: str,
    severity: str,
    symptoms: str,
    affected_users: str = "unknown"
) -> list[types.PromptMessage]:
    """Create infrastructure incident response prompt."""
    text = f"""**INCIDENT RESPONSE PROTOCOL**

**Incident Details:**
- Service: {service_name}
- Severity: {severity}
- Symptoms: {symptoms}
- Affected Users: {affected_users}

**Immediate Response Plan:**

**1. ASSESS (First 5 minutes)**
- Determine incident scope and impact
- Check service dependencies
- Verify monitoring alerts and metrics
- Estimate affected user count

**2. COMMUNICATE (Within 10 minutes)**
- Notify incident response team
- Update status page if customer-facing
- Set up incident communication channel
- Establish incident commander

**3. INVESTIGATE (Ongoing)**
- Analyze logs and metrics for {service_name}
- Check recent deployments and changes
- Review infrastructure health
- Identify potential root causes

**4. MITIGATE (Priority Actions)**
- Implement immediate workarounds
- Scale resources if capacity-related
- Rollback recent changes if applicable
- Activate backup systems if needed

**5. RESOLVE (Long-term)**
- Apply permanent fixes
- Verify full service restoration
- Update monitoring and alerting
- Schedule post-incident review

Please provide specific investigation steps and mitigation strategies for this {severity} incident."""
    
    return [types.PromptMessage(role="user", content=types.TextContent(type="text", text=text))]

def create_monitoring_strategy_design_prompt(
    system_type: str,
    scale: str,
    critical_metrics: str,
    budget_tier: str = "medium"
) -> list[types.PromptMessage]:
    """Create monitoring strategy design prompt."""
    text = f"""Design a comprehensive monitoring strategy for this system:

**System Information:**
- Type: {system_type}
- Scale: {scale}
- Critical Metrics: {critical_metrics}
- Budget Tier: {budget_tier}

**Monitoring Strategy Design:**

**1. OBSERVABILITY PILLARS**
- **Metrics**: Define key performance indicators
- **Logs**: Structured logging strategy
- **Traces**: Distributed tracing approach

**2. MONITORING LAYERS**
- **Infrastructure**: Hardware, network, storage
- **Platform**: Containers, orchestration, middleware
- **Application**: Business logic, user experience
- **Business**: KPIs and business metrics

**3. ALERTING STRATEGY**
- **Critical Alerts**: Immediate response required
- **Warning Alerts**: Proactive monitoring
- **Info Alerts**: Awareness and trending

**4. DASHBOARD DESIGN**
- **Executive Dashboard**: High-level business metrics
- **Operations Dashboard**: Real-time system health
- **Developer Dashboard**: Application performance

**5. TOOL RECOMMENDATIONS**
- Monitoring tools for {budget_tier} budget
- Integration approaches
- Data retention policies

Please provide specific tool recommendations, alert thresholds, and implementation timeline."""
    
    return [types.PromptMessage(role="user", content=types.TextContent(type="text", text=text))]

# =============================================================================
# SERVER SETUP
# =============================================================================

@click.command()
@click.option("--port", default=8001, help="Port to listen on for SSE")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
def main(port: int, transport: str) -> int:
    """Start the DevOps Prompts MCP server."""
    app = Server("devops-prompts-server")

    @app.list_prompts()
    async def list_prompts() -> list[types.Prompt]:
        """List all available DevOps prompts."""
        return [
            types.Prompt(
                name="cicd_pipeline_analysis",
                title="CI/CD Pipeline Analysis",
                description="Analyze CI/CD pipeline failures and provide troubleshooting guidance",
                arguments=[
                    types.PromptArgument(name="pipeline_name", description="Name of the pipeline", required=True),
                    types.PromptArgument(name="failure_stage", description="Stage where failure occurred", required=True),
                    types.PromptArgument(name="error_logs", description="Error logs from the failure", required=True),
                    types.PromptArgument(name="tech_stack", description="Technology stack used", required=False),
                ],
            ),
            types.Prompt(
                name="deployment_strategy_review",
                title="Deployment Strategy Review",
                description="Review and optimize deployment strategies for applications",
                arguments=[
                    types.PromptArgument(name="application_type", description="Type of application", required=True),
                    types.PromptArgument(name="current_strategy", description="Current deployment strategy", required=True),
                    types.PromptArgument(name="environment", description="Target environment", required=False),
                    types.PromptArgument(name="traffic_volume", description="Expected traffic volume", required=False),
                ],
            ),
            types.Prompt(
                name="code_review_checklist",
                title="Code Review Checklist",
                description="Generate comprehensive code review checklist for different scenarios",
                arguments=[
                    types.PromptArgument(name="language", description="Programming language", required=True),
                    types.PromptArgument(name="change_type", description="Type of code change", required=True),
                    types.PromptArgument(name="security_level", description="Security level requirements", required=False),
                ],
            ),
            types.Prompt(
                name="infrastructure_incident_response",
                title="Infrastructure Incident Response",
                description="Guide incident response for infrastructure issues",
                arguments=[
                    types.PromptArgument(name="service_name", description="Name of affected service", required=True),
                    types.PromptArgument(name="severity", description="Incident severity level", required=True),
                    types.PromptArgument(name="symptoms", description="Observed symptoms", required=True),
                    types.PromptArgument(name="affected_users", description="Number of affected users", required=False),
                ],
            ),
            types.Prompt(
                name="monitoring_strategy_design",
                title="Monitoring Strategy Design",
                description="Design comprehensive monitoring strategy for systems",
                arguments=[
                    types.PromptArgument(name="system_type", description="Type of system to monitor", required=True),
                    types.PromptArgument(name="scale", description="System scale", required=True),
                    types.PromptArgument(name="critical_metrics", description="Critical metrics to track", required=True),
                    types.PromptArgument(name="budget_tier", description="Budget tier", required=False),
                ],
            ),
        ]

    @app.get_prompt()
    async def get_prompt(name: str, arguments: dict[str, str] | None = None) -> types.GetPromptResult:
        """Get a specific prompt by name."""
        if arguments is None:
            arguments = {}

        if name == "cicd_pipeline_analysis":
            messages = create_cicd_pipeline_analysis_prompt(
                pipeline_name=arguments.get("pipeline_name", ""),
                failure_stage=arguments.get("failure_stage", ""),
                error_logs=arguments.get("error_logs", ""),
                tech_stack=arguments.get("tech_stack", "Docker, Kubernetes, Jenkins")
            )
            return types.GetPromptResult(
                messages=messages,
                description="CI/CD pipeline failure analysis and troubleshooting guidance"
            )
        elif name == "deployment_strategy_review":
            messages = create_deployment_strategy_review_prompt(
                application_type=arguments.get("application_type", ""),
                current_strategy=arguments.get("current_strategy", ""),
                environment=arguments.get("environment", "production"),
                traffic_volume=arguments.get("traffic_volume", "medium")
            )
            return types.GetPromptResult(
                messages=messages,
                description="Deployment strategy review and optimization recommendations"
            )
        elif name == "code_review_checklist":
            messages = create_code_review_checklist_prompt(
                language=arguments.get("language", ""),
                change_type=arguments.get("change_type", ""),
                security_level=arguments.get("security_level", "standard")
            )
            return types.GetPromptResult(
                messages=messages,
                description="Comprehensive code review checklist"
            )
        elif name == "infrastructure_incident_response":
            messages = create_infrastructure_incident_response_prompt(
                service_name=arguments.get("service_name", ""),
                severity=arguments.get("severity", ""),
                symptoms=arguments.get("symptoms", ""),
                affected_users=arguments.get("affected_users", "unknown")
            )
            return types.GetPromptResult(
                messages=messages,
                description="Infrastructure incident response protocol"
            )
        elif name == "monitoring_strategy_design":
            messages = create_monitoring_strategy_design_prompt(
                system_type=arguments.get("system_type", ""),
                scale=arguments.get("scale", ""),
                critical_metrics=arguments.get("critical_metrics", ""),
                budget_tier=arguments.get("budget_tier", "medium")
            )
            return types.GetPromptResult(
                messages=messages,
                description="Comprehensive monitoring strategy design"
            )
        else:
            raise ValueError(f"Unknown prompt: {name}")

    logger.info("Starting DevOps Prompts MCP Server...")
    logger.info("Available prompts:")
    logger.info("  - cicd_pipeline_analysis: Analyze CI/CD pipeline failures")
    logger.info("  - deployment_strategy_review: Review deployment strategies")
    logger.info("  - code_review_checklist: Generate code review checklists")
    logger.info("  - infrastructure_incident_response: Guide incident response")
    logger.info("  - monitoring_strategy_design: Design monitoring strategies")

    if transport == "sse":
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.responses import Response
        from starlette.routing import Mount, Route

        sse = SseServerTransport("/messages/")

        async def handle_sse(request: Request):
            async with sse.connect_sse(request.scope, request.receive, request._send) as streams:  # type: ignore[reportPrivateUsage]
                await app.run(streams[0], streams[1], app.create_initialization_options())
            return Response()

        starlette_app = Starlette(
            debug=True,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        import uvicorn
        logger.info(f"Starting SSE server on http://127.0.0.1:{port}/sse")
        uvicorn.run(starlette_app, host="127.0.0.1", port=port)
    else:
        from mcp.server.stdio import stdio_server

        async def arun():
            async with stdio_server() as streams:
                await app.run(streams[0], streams[1], app.create_initialization_options())

        logger.info("Starting stdio server...")
        anyio.run(arun)

    return 0

if __name__ == "__main__":
    main()