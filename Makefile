# Makefile

# Set Bash as the default shell
SHELL:=/bin/bash

# Variables
PROJECT_CONFIG_FILES = lab/processes/config.ini
TEMPLATE_PROJECT_NAME = kdl-project-template
DEFAULT = n
OLD_PROJECT_NAME = $(shell basename $$PWD)
SECRET_NAME = mlflow-server-secret

BOLD_GREEN_IN=\033[1;32m
BOLD_RED_IN=\033[1;31m
BOLD_YELLOW_IN=\033[1;33m
STOP=\033[0m
DRONE_FILE = .drone.yml
DRONE_TEMPLATE_PROJECT_NAME = project-template
SONAR_FILE = sonar-project.properties

.ONESHELL:
setup_project:
	@printf "Setting up the project...\n\n"

# Ask for the right project ID
		echo -e "The project name is set to $(BOLD_GREEN_IN)$(OLD_PROJECT_NAME)$(STOP) by default."
		read -p "Is this the correct KAI Lab project ID shown in the Settings tab? [Y/n]: " DEFAULT

		if [ "$${DEFAULT:-Y}" = "y" ] || [ "$${DEFAULT:-Y}" = "Y" ]; then \
			echo -e "\n$(BOLD_GREEN_IN)✓ Using $(OLD_PROJECT_NAME) as the project name!$(STOP)\n"; \
		else \

# If not correct by default, ask for the correct one
			read -p "Please enter the project ID (): " NEW; \
			LOWER_NEW=$(shell echo $${NEW} | tr A-Z a-z)
			printf "\n$(BOLD_GREEN_IN)✓ Using $${LOWER_NEW} as the project name!$(STOP)\n\n";

# and update config files
			sed -si "s/$(TEMPLATE_PROJECT_NAME)/$${NEW}/g" $(PROJECT_CONFIG_FILES)
			if [ $$? -eq 0 ]; then \
				echo -e "$(BOLD_GREEN_IN)✓ All project references where updated to $${NEW} in $(PROJECT_CONFIG_FILES)!$(STOP)"; \
			else \
				"$(BOLD_RED_IN)× There was an error updating the references in the `$(PROJECT_CONFIG_FILES)` file, please check for all `$(TEMPLATE_PROJECT_NAME)` references and replace them with your project name!$(STOP)"; \
			fi

# and update the URLs in the Drone file
			sed -si "s/$(DRONE_TEMPLATE_PROJECT_NAME)/$${NEW}/g" $(DRONE_FILE)
			if [ $$? -eq 0 ]; then \
				echo -e "$(BOLD_GREEN_IN)✓ All project references where updated to $${NEW} in $(DRONE_FILE)!$(STOP)"; \
			else \
				echo -e "$(BOLD_RED_IN)× There was an error updating the references in the `$(PROJECT_CONFIG_FILES)` file, please check for all `$(TEMPLATE_PROJECT_NAME)` references and replace them with your project name!$(STOP)"; \
			fi

# and the Secrets in the Drone file
			sed -si "s/$(SECRET_NAME)/$${NEW}-mlflow-secret/g" $(DRONE_FILE)
			if [ $$? -eq 0 ]; then \
				echo -e "$(BOLD_GREEN_IN)✓ All Secrets references where updated to $${NEW}-mlflow-secret in $(DRONE_FILE)!$(STOP)"; \
			else \
				echo -e "$(BOLD_RED_IN)× There was an error updating the Secret references in the `$(DRONE_FILE)` file, please check for all `$(TEMPLATE_PROJECT_NAME)` references and replace them with your project name!$(STOP)"; \
			fi

			echo ""

# ask if the user wants to commit the changes
			git add $(PROJECT_CONFIG_FILES) $(DRONE_FILE)
			git commit -m "Setup project"

			read -p "Do you want to push the changes to the repository? [Y/n]: " PUSH
			if [ "$${PUSH:-Y}" = "y" ] || [ "$${PUSH:-Y}" = "Y" ]; then \
				git push && echo "\033[32m✓ Changes have been successfully pushed to the repository!\033[0m"; \
			else \
				echo "\033[33m⚠ Please remember to push the changes to the repository!\033[0m"; \
			fi
		fi
	
# extra warnings
		echo -e "\n$(BOLD_YELLOW_IN)⚠ Please remember to update the $(SONAR_FILE) file with your project name and key, and rename the project name in the README file!$(STOP)"
		echo -e "$(BOLD_YELLOW_IN)⚠ Watch out for the Volume Claim name in the Drone file!$(STOP)"