@description('Primary location for all resources. Default swedencentral: one of only two regions (with eastus2) where Anthropic Claude models can be deployed, and where gpt-5.4 + text-embedding-3-small are available via GlobalStandard/DataZone deployments (NOT via regional Standard — hence the embedding deployment below uses GlobalStandard). switzerlandnorth remains valid for an OpenAI-only stack but cannot host the Claude judge model.')
@allowed([
  'swedencentral'
  'eastus2'
  'switzerlandnorth'
  'westus'
  'japaneast'
  'australiaeast'
  'uaenorth'
  'canadaeast'
  'eastus'
])
param location string = 'swedencentral'

@description('Base name for resources')
param baseName string = 'ragpipe'

@description('Chat model deployment name (generator; also the RAGAS judge until the judge-model split lands).')
param chatModel string = 'gpt-5.4'

@description('Chat model version. gpt-5.4 GA version is 2026-03-05.')
param chatModelVersion string = '2026-03-05'

@description('Judge model deployment name. claude-sonnet-4-6 (preview) is an Anthropic partner model: deployable only when the Foundry account is in swedencentral or eastus2, and the subscription needs Azure Marketplace access with pay-as-you-go billing (first-time deployments may require a one-time marketplace offer acceptance in the portal). Set to empty string to skip the deployment.')
param judgeModel string = 'claude-sonnet-4-6'

@description('Offline RAGAS judge deployment. DeepSeek-V4-Pro (preview) is sold directly by Azure: GlobalStandard in all regions (incl. swedencentral), served on the OpenAI-compatible route, Azure-direct licensing — no marketplace acceptance needed (unlike Claude). Third family besides the OpenAI generator and the Anthropic online gate (ADR-0009). Set to empty string to skip.')
param offlineJudgeModel string = 'DeepSeek-V4-Pro'

@description('DeepSeek offline-judge deployment version. Required: unlike the OpenAI/Anthropic deployments (which resolve a current default when version is omitted), the DeepSeek route rejects a version-less deployment with DeploymentModelNotSupported. GA version in swedencentral is 2026-04-23.')
param offlineJudgeModelVersion string = '2026-04-23'

@description('Embedding model deployment name. Deployed as GlobalStandard: text-embedding-3-small has no regional-Standard availability in swedencentral (that limitation is what previously forced switzerlandnorth).')
param embeddingModel string = 'text-embedding-3-small'

@description('Object (principal) ID to grant data-plane access to the Foundry models and the Search index. With azd this is auto-populated from AZURE_PRINCIPAL_ID (the deploying user or CI service principal). Leave empty to skip role assignments (e.g. when assigning them out of band).')
param principalId string = ''

@description('Type of the principal receiving the role assignments: User for a developer running azd locally, ServicePrincipal for CI / managed identity.')
@allowed([
  'User'
  'ServicePrincipal'
])
param principalType string = 'User'

// Well-known built-in role definition IDs (control-plane stable GUIDs).
// Cognitive Services OpenAI User grants the data actions for the `/openai` route used by
// both generation (FoundryAgent) and the OpenAI-compatible embedding calls. Our embedding
// model is an Azure OpenAI deployment, served only on `/openai` (not the `/models`
// inference route), so this role is sufficient — no Cognitive Services User role needed.
var cognitiveServicesOpenAIUserRoleId = '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'
var searchServiceContributorRoleId = '7ca78c08-252a-4471-8644-bb5ff32d4ba0'
var searchIndexDataContributorRoleId = '8ebe5a00-799e-43f5-93ac-243d3dce84a7'

resource search 'Microsoft.Search/searchServices@2024-06-01-preview' = {
  name: '${baseName}-search'
  location: location
  sku: { name: 'standard' }
  properties: {
    semanticSearch: 'standard'
    replicaCount: 1
    partitionCount: 1
    // Accept Microsoft Entra tokens (RBAC) as well as API keys. Without this the
    // service is apiKeyOnly and rejects our DefaultAzureCredential auth with 403.
    disableLocalAuth: false
    authOptions: {
      aadOrApiKey: {
        aadAuthFailureMode: 'http401WithBearerChallenge'
      }
    }
  }
}

resource foundry 'Microsoft.CognitiveServices/accounts@2025-04-01-preview' = {
  name: '${baseName}-foundry'
  location: location
  kind: 'AIServices'
  sku: { name: 'S0' }
  // A system-assigned managed identity is required for Foundry project creation.
  identity: { type: 'SystemAssigned' }
  properties: {
    allowProjectManagement: true
    customSubDomainName: '${baseName}-foundry'
  }
}

resource project 'Microsoft.CognitiveServices/accounts/projects@2025-04-01-preview' = {
  parent: foundry
  name: '${baseName}-project'
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {}
}

// capacity 100 gives the chat model enough RPM/TPM for the generator agent AND the
// RAGAS LLM-judge (offline eval makes many concurrent judge calls; capacity 10
// throttled the per-stage sweep to 20s+/step). Free on GlobalStandard — billed per token.
resource chat 'Microsoft.CognitiveServices/accounts/deployments@2025-04-01-preview' = {
  parent: foundry
  name: chatModel
  sku: { name: 'GlobalStandard', capacity: 100 }
  properties: {
    model: { format: 'OpenAI', name: chatModel, version: chatModelVersion }
  }
}

// capacity sets the deployment's rate limit (≈ capacity req/10s and capacity*1000
// tokens/min). The default of 10 is far too low for bulk corpus ingestion (every
// worker thrashes on 429s); 120 lets ~580 pages ingest in minutes. GlobalStandard
// (required for 3-small in swedencentral) bills per token consumed, not per
// capacity, so this is free headroom.
resource embedding 'Microsoft.CognitiveServices/accounts/deployments@2025-04-01-preview' = {
  parent: foundry
  name: embeddingModel
  dependsOn: [chat]
  sku: { name: 'GlobalStandard', capacity: 120 }
  properties: {
    model: { format: 'OpenAI', name: embeddingModel }
  }
}

// Anthropic partner model for the RAGAS judge (deployed now, wired up in the
// judge-independence change). Claude models support only GlobalStandard and are
// marketplace-billed; version is omitted so Azure assigns the current default.
// Deployments on one account must be created sequentially — hence dependsOn.
resource judge 'Microsoft.CognitiveServices/accounts/deployments@2025-04-01-preview' = if (!empty(judgeModel)) {
  parent: foundry
  name: judgeModel
  dependsOn: [embedding]
  sku: { name: 'GlobalStandard', capacity: 50 }
  properties: {
    model: { format: 'Anthropic', name: judgeModel }
  }
}

// DeepSeek model (sold directly by Azure) for the OFFLINE RAGAS judge (ADR-0009):
// a third family so offline scores are independent of both the generator (OpenAI)
// and the online gate (Anthropic). Sequential dependsOn — same
// one-deployment-at-a-time rule.
resource offlineJudge 'Microsoft.CognitiveServices/accounts/deployments@2025-04-01-preview' = if (!empty(offlineJudgeModel)) {
  parent: foundry
  name: offlineJudgeModel
  // Sequential one-deployment-at-a-time chain that survives judgeModel='' —
  // a dependsOn entry naming a skipped conditional resource produces a
  // malformed (empty-name) resource ID at ARM validation time.
  dependsOn: empty(judgeModel) ? [embedding] : [judge]
  // 50 matches the online judge: the offline RAGAS sweep runs many concurrent
  // judge calls, but per-item scoring is bounded by the harness's sequential
  // replay, so chat-level (100) headroom isn't needed.
  sku: { name: 'GlobalStandard', capacity: 50 }
  properties: {
    model: { format: 'DeepSeek', name: offlineJudgeModel, version: offlineJudgeModelVersion }
  }
}

// --- Data-plane RBAC for the deploying principal -------------------------------
// Control-plane Owner does NOT grant data-plane access. These assignments let the
// principal call the embedding/chat deployments and read/write the Search index,
// which the post-provision ingestion + agent-registration steps require.

resource openAIUserAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(principalId)) {
  name: guid(foundry.id, principalId, cognitiveServicesOpenAIUserRoleId)
  scope: foundry
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', cognitiveServicesOpenAIUserRoleId)
    principalId: principalId
    principalType: principalType
  }
}

resource searchServiceContributorAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(principalId)) {
  name: guid(search.id, principalId, searchServiceContributorRoleId)
  scope: search
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', searchServiceContributorRoleId)
    principalId: principalId
    principalType: principalType
  }
}

resource searchIndexDataContributorAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(principalId)) {
  name: guid(search.id, principalId, searchIndexDataContributorRoleId)
  scope: search
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', searchIndexDataContributorRoleId)
    principalId: principalId
    principalType: principalType
  }
}

output FOUNDRY_PROJECT_ENDPOINT string = 'https://${foundry.name}.services.ai.azure.com/api/projects/${project.name}'
output SEARCH_ENDPOINT string = 'https://${search.name}.search.windows.net'
output FOUNDRY_CHAT_MODEL string = chatModel
output FOUNDRY_EMBEDDING_MODEL string = embeddingModel
output JUDGE_MODEL string = judgeModel
output OFFLINE_JUDGE_MODEL string = offlineJudgeModel
