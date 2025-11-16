terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~>3.70"
    }
  }
}

provider "azurerm" {
  features {}
}

data "azurerm_resource_group" "lab" {
  name = var.resource_group_name
}

resource "azurerm_container_registry" "acr" {
  name                     = var.acr_name
  resource_group_name      = data.azurerm_resource_group.lab.name
  location                 = data.azurerm_resource_group.lab.location
  sku                      = "Basic"
  admin_enabled            = true
}

resource "azurerm_service_plan" "plan" {
  name                = var.plan_name
  location            = data.azurerm_resource_group.lab.location
  resource_group_name = data.azurerm_resource_group.lab.name
  os_type             = "Linux"
  sku_name            = "B1"
}

resource "azurerm_linux_web_app" "flask" {
  name                = var.app_service_name
  resource_group_name = data.azurerm_resource_group.lab.name
  location            = data.azurerm_resource_group.lab.location
  service_plan_id     = azurerm_service_plan.plan.id

  site_config {}

  app_settings = {
    "WEBSITES_PORT" = "8000"
  }

  https_only = true
}

