provider "aws" {
  region = var.aws_region
}

terraform {
  backend "s3" {
    # bucket 名通过 CI/CD 的 -backend-config="bucket=..." 动态传入
    # 本地调试: terraform init -backend-config="bucket=fortune-teller-terraform-state-<account_id>"
    key          = "production/terraform.tfstate"
    region       = "us-east-1"
    use_lockfile = true
    encrypt      = true
  }
}

import {
  to = module.ecr.aws_ecr_repository.fortune_api
  id = "${var.project_name}-api"
}

import {
  to = module.ecr.aws_ecr_repository.fortune_app
  id = "${var.project_name}-app"
}

module "ecr" {
  source = "../../modules/ecr"

  api_repository_name = "${var.project_name}-api"
  app_repository_name = "${var.project_name}-app"
}

module "networking" {
  source = "../../modules/networking"

  project_name           = var.project_name
  vpc_cidr              = var.vpc_cidr
  public_subnet_cidrs   = var.public_subnet_cidrs
  availability_zones    = var.availability_zones
  alb_security_group_ids = [module.alb.alb_security_group_id]
}

module "alb" {
  source = "../../modules/alb"

  project_name      = var.project_name
  vpc_id           = module.networking.vpc_id
  public_subnet_ids = module.networking.public_subnet_ids
}

module "ecs" {
  source = "../../modules/ecs"

  project_name                     = var.project_name
  api_image_url                    = module.ecr.api_repository_url
  app_image_url                    = module.ecr.app_repository_url
  subnet_ids                       = module.networking.public_subnet_ids
  security_group_id                = module.networking.security_group_id
  aws_region                       = var.aws_region
  service_discovery_namespace_id   = module.networking.service_discovery_namespace_id
  service_discovery_namespace_name = module.networking.service_discovery_namespace_name
  api_target_group_arn             = module.alb.api_target_group_arn
  app_target_group_arn             = module.alb.app_target_group_arn
  # Image tags to control deploys; set via CI/CD or manual tfvars
  api_image_tag                    = var.api_image_tag
  app_image_tag                    = var.app_image_tag
  # Route frontend API calls through ALB public DNS with path prefix
  app_api_url                      = "http://${module.alb.alb_dns_name}/api"
  # Increase API resources to reduce OOM kills
  api_cpu                          = var.api_cpu
  api_memory                       = var.api_memory
  api_environment_variables = [
    {
      name  = "MOONSHOT_API_KEY"
      value = var.moonshot_api_key
    },
    {
      name  = "API_ROOT_PATH"
      value = "/api"
    },
    {
      name  = "DEBUG"
      value = "true"
    }
  ]
  app_environment_variables = []
}