variable "aws_region" {
  default = "eu-north-1"  # Stockholm region
}

variable "ami_id" {
  default = "ami-006a6296aa17e4546"  # Ubuntu 20.04 in eu-north-1
}

variable "instance_type" {
  default = "t3.micro"
}

variable "key_name" {
  default = "berna-key"
}

variable "private_key_path" {
  default = "~/.ssh/berna-key.pem"
}