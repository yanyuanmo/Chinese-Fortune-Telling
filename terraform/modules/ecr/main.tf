import {
  to = aws_ecr_repository.fortune_api
  id = var.api_repository_name
}

resource "aws_ecr_repository" "fortune_api" {
  name                 = var.api_repository_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

import {
  to = aws_ecr_repository.fortune_app
  id = var.app_repository_name
}

resource "aws_ecr_repository" "fortune_app" {
  name                 = var.app_repository_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}