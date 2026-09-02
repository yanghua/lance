if [ -d "output" ]; then
    rm -r "output"
fi
export PATH=/root/.cargo/bin/:$JAVA_HOME/bin:/opt/apache-maven-${MAVEN_VERSION}/bin:$PATH

# 条件化 Rust 版本更新
if [ -n "${CUSTOM_LANCE_VERSION}" ]; then
    echo "[1/5] Updating Rust packages to ${CUSTOM_LANCE_VERSION}"
    for pkg in $(cargo metadata --format-version 1 |
      jq -r '.packages[] | select( [.id] | inside([$ARGS.positional[]]) ) | .name' \
      --args $(cargo metadata --format-version 1 | jq -r '.workspace_members[]')); do
      if [[ "$pkg" != "object_store" ]]; then
        cargo set-version -p "$pkg" "${CUSTOM_LANCE_VERSION}"
      fi
    done
else
    echo "[1/5] Skipping Rust version update"
fi


mkdir output/
cd python
if [ -n "${CUSTOM_LANCE_VERSION}" ]; then
    echo "[2/5] Setting Python package version"
    cargo set-version "${CUSTOM_LANCE_VERSION}"
fi

echo "[3/5] Building Python wheel"
maturin build --release

if ls target/wheels/ve_pylance*.whl 1> /dev/null 2>&1; then
    cp target/wheels/ve_pylance*.whl ../output/
    if [ "$CUSTOM_TYPE" = "release" ]; then
        echo "[4/5] Uploading Python wheel to PyPI"
        python3 -m twine upload \
        --username __token__ \
        --password ${TWINE_PASSWORD} \
         ../output/ve_pylance*.whl
    else
        echo "[4/5] Skipping PyPI upload (CUSTOM_TYPE is not 'release')"
    fi
else
    echo "No ve_pylance file found, skipping copy."
fi

if ls target/wheels/pylance-*.whl 1> /dev/null 2>&1; then
  cp target/wheels/pylance-*.whl ../output/
else
   echo "No pylance file found, skipping copy."
fi

cd ../java
if [ -n "${CUSTOM_LANCE_VERSION}" ]; then
    echo "[4/5] Updating Java project version"
    mvn -B -DnewVersion="${CUSTOM_LANCE_VERSION}" -DprocessAllModules=true versions:set
    mvn -B versions:commit
fi
# 动态构建参数
mvn_args=(
  "clean"
  "package"
  "-B"
  "-DskipTests"
  "-Drust.release.build=true"
)
[ -n "${CUSTOM_LANCE_VERSION}" ] && mvn_args+=("-Drevision=${CUSTOM_LANCE_VERSION}")

echo "[5/5] Building Java artifacts"
mvn "${mvn_args[@]}"

output_dir="../output"
# 定义复制函数（安全检查）
safe_copy() {
    local pattern="$1"
    # 检查文件是否存在
    if ls $pattern >/dev/null 2>&1; then
        echo "复制文件: $pattern"
        cp $pattern "$output_dir"
    else
        echo "文件不存在: $pattern - 跳过"
    fi
}

# 复制 Spark JAR
safe_copy "spark/target/lance-spark-*[0-9].jar"

# 复制 Core JAR
safe_copy "core/target/lance-core-*[0-9].jar"

# 复制 Catalog JAR
safe_copy "catalog/target/lance-catalog-*[0-9].jar"

echo "操作完成"
