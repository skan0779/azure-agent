locals {
  base_name     = "${var.project_name}-${var.environment}"
  dashed_suffix = var.name_suffix != "" ? "-${var.name_suffix}" : ""

  names = {
    vnet              = "${local.base_name}${local.dashed_suffix}-langfuse-vnet"
    subnet            = "${local.base_name}${local.dashed_suffix}-langfuse-snet"
    nsg               = "${local.base_name}${local.dashed_suffix}-langfuse-nsg"
    public_ip         = "${local.base_name}${local.dashed_suffix}-langfuse-pip"
    network_interface = "${local.base_name}${local.dashed_suffix}-langfuse-nic"
    vm                = "${local.base_name}${local.dashed_suffix}-langfuse-vm"
  }

  ssh_public_key = var.ssh_public_key != null ? var.ssh_public_key : file(pathexpand(var.ssh_public_key_path))
  tags           = merge(var.tags, { Component = "langfuse-vm" })
}
