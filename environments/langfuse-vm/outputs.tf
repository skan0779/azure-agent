output "vm_name" {
  description = "Langfuse VM name."
  value       = azurerm_linux_virtual_machine.main.name
}

output "vm_public_ip" {
  description = "Langfuse VM public IP address."
  value       = azurerm_public_ip.main.ip_address
}

output "ssh_command" {
  description = "SSH command for the Langfuse VM."
  value       = "ssh ${var.admin_username}@${azurerm_public_ip.main.ip_address}"
}

output "langfuse_url" {
  description = "Langfuse URL to use as LANGFUSE_BASE_URL after Docker Compose is running."
  value       = "http://${azurerm_public_ip.main.ip_address}:3000"
}
