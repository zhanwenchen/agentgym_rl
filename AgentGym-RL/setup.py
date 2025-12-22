# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# setup.py is the fallback installation script when pyproject.toml does not work
from setuptools import setup, find_packages
import os

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

# Use a default version since verl is being removed
__version__ = '0.1.0'

install_requires = [
  'numpy',
  'pandas',
  'transformers',
]

TEST_REQUIRES = ['pytest']
GPU_REQUIRES = []

extras_require = {
  'test': TEST_REQUIRES,
  'gpu': GPU_REQUIRES,
}

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='agentgym-rl',
    version=__version__,
    package_dir={'': '.'},
    packages=find_packages(where='.'),
    url='https://github.com/zhanwenchen/agentgym_rl',
    license='Apache 2.0',
    author='AgentGym-RL Contributors',
    author_email='',
    description='AgentGym-RL: Minimal pipeline for agent training',
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={'': ['version/*']},
    include_package_data=True,
    long_description=long_description,
    long_description_content_type='text/markdown'
)