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

resource search 'Microsoft.Search/searchServices@2024-06-01-preview' = {
  name: '${baseName}-search'
  location: location
  sku: { name: 'standard' }
  properties: {
    semanticSearch: 'standard'
    replicaCount: 1
    partitionCount: 1
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

output FOUNDRY_PROJECT_ENDPOINT string = 'https://${foundry.name}.services.ai.azure.com/api/projects/${project.name}'
output SEARCH_ENDPOINT string = 'https://${search.name}.search.windows.net'
output FOUNDRY_CHAT_MODEL string = chatModel
output FOUNDRY_EMBEDDING_MODEL string = embeddingModel
