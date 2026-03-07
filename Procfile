# ==============================================================================
# PROCFILE - Process definitions for deployment
# ==============================================================================
# Railway and Heroku read this file to know how to start the application.
#
# "web" is the process type that receives HTTP traffic.
# $PORT is automatically set by Railway to the port it expects traffic on.
#
# To run locally with this Procfile:
#   pip install honcho
#   honcho start
# ==============================================================================

web: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
