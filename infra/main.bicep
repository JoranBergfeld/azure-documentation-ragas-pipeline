@description('Primary location for all resources. Restricted to regions where gpt-4o (2024-11-20), text-embedding-3-small, and Azure AI Search (with semantic ranker) are all available on the Standard deployment type. Note: text-embedding-3-small is NOT available in most EU regions; switzerlandnorth is the only EU option.')
@allowed([
  'switzerlandnorth'
  'westus'
  'japaneast'
  'australiaeast'
  'uaenorth'
  'canadaeast'
  'eastus'
  'eastus2'
])
param location string = 'switzerlandnorth'

@description('Base name for resources')
param baseName string = 'ragpipe'

@description('Chat model deployment name')
param chatModel string = 'gpt-4o'

@description('Embedding model deployment name. text-embedding-3-small must be deployed in one of the allowed regions above (it is not available in swedencentral).')
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

resource chat 'Microsoft.CognitiveServices/accounts/deployments@2025-04-01-preview' = {
  parent: foundry
  name: chatModel
  sku: { name: 'GlobalStandard', capacity: 10 }
  properties: {
    model: { format: 'OpenAI', name: chatModel }
  }
}

resource embedding 'Microsoft.CognitiveServices/accounts/deployments@2025-04-01-preview' = {
  parent: foundry
  name: embeddingModel
  dependsOn: [chat]
  sku: { name: 'Standard', capacity: 10 }
  properties: {
    model: { format: 'OpenAI', name: embeddingModel }
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
