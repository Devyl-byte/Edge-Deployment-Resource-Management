FROM python:3.12-slim

# 2. Set the directory where your code will live inside the container
WORKDIR /app

# 3. Copy only the dependency files first (this makes builds faster)
COPY pyproject.toml uv.lock ./

# 4. Install 'uv' and use it to install your project's dependencies
RUN pip install --no-cache-dir uv && \
    uv pip install --system --requirement pyproject.toml

# 5. Copy the rest of your project files into the container
COPY . .

# 6. Tell Docker to run the "server" command you defined earlier
EXPOSE 7860
CMD ["python", "server/app.py"]