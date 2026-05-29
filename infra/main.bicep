@description('Primary location for all resources')
param location string = resourceGroup().location

@description('Base name for resources')
param baseName string = 'ragpipe'

@description('Chat model deployment name')
param chatModel string = 'gpt-4o'

@description('Embedding model deployment name')
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
  properties: {
    allowProjectManagement: true
    customSubDomainName: '${baseName}-foundry'
  }
}

resource project 'Microsoft.CognitiveServices/accounts/projects@2025-04-01-preview' = {
  parent: foundry
  name: '${baseName}-project'
  location: location
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
