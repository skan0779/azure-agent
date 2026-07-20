data "azurerm_resource_group" "main" {
  name = var.resource_group_name
}

resource "azurerm_virtual_network" "main" {
  name                = local.names.vnet
  resource_group_name = data.azurerm_resource_group.main.name
  location            = data.azurerm_resource_group.main.location
  address_space       = var.vnet_address_space
  tags                = local.tags
}

resource "azurerm_subnet" "main" {
  name                 = local.names.subnet
  resource_group_name  = data.azurerm_resource_group.main.name
  virtual_network_name = azurerm_virtual_network.main.name
  address_prefixes     = var.subnet_address_prefixes
}

resource "azurerm_network_security_group" "main" {
  name                = local.names.nsg
  resource_group_name = data.azurerm_resource_group.main.name
  location            = data.azurerm_resource_group.main.location
  tags                = local.tags
}

resource "azurerm_network_security_rule" "ssh" {
  name                        = "AllowSshFromClientIp"
  priority                    = 100
  direction                   = "Inbound"
  access                      = "Allow"
  protocol                    = "Tcp"
  source_port_range           = "*"
  destination_port_range      = "22"
  source_address_prefixes     = var.allowed_source_ip_ranges
  destination_address_prefix  = "*"
  resource_group_name         = data.azurerm_resource_group.main.name
  network_security_group_name = azurerm_network_security_group.main.name
}

resource "azurerm_network_security_rule" "langfuse" {
  name                        = "AllowLangfuseFromClientIp"
  priority                    = 110
  direction                   = "Inbound"
  access                      = "Allow"
  protocol                    = "Tcp"
  source_port_range           = "*"
  destination_port_range      = "3000"
  source_address_prefixes     = var.allowed_source_ip_ranges
  destination_address_prefix  = "*"
  resource_group_name         = data.azurerm_resource_group.main.name
  network_security_group_name = azurerm_network_security_group.main.name
}

resource "azurerm_subnet_network_security_group_association" "main" {
  subnet_id                 = azurerm_subnet.main.id
  network_security_group_id = azurerm_network_security_group.main.id
}

resource "azurerm_public_ip" "main" {
  name                = local.names.public_ip
  resource_group_name = data.azurerm_resource_group.main.name
  location            = data.azurerm_resource_group.main.location
  allocation_method   = "Static"
  sku                 = "Standard"
  tags                = local.tags
}

resource "azurerm_network_interface" "main" {
  name                = local.names.network_interface
  resource_group_name = data.azurerm_resource_group.main.name
  location            = data.azurerm_resource_group.main.location
  tags                = local.tags

  ip_configuration {
    name                          = "default"
    subnet_id                     = azurerm_subnet.main.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.main.id
  }
}

resource "azurerm_linux_virtual_machine" "main" {
  name                            = local.names.vm
  resource_group_name             = data.azurerm_resource_group.main.name
  location                        = data.azurerm_resource_group.main.location
  size                            = var.vm_size
  admin_username                  = var.admin_username
  disable_password_authentication = true
  network_interface_ids           = [azurerm_network_interface.main.id]
  tags                            = local.tags

  admin_ssh_key {
    username   = var.admin_username
    public_key = local.ssh_public_key
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Premium_LRS"
    disk_size_gb         = var.os_disk_size_gb
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "ubuntu-24_04-lts"
    sku       = "server"
    version   = "latest"
  }
}
