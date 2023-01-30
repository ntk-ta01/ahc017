入力生成

```bash
cargo run -p tools --bin gen ./tools/seeds.txt --dir=./tools/in
```

テスト

```bash
cargo run -p tools --bin tester
```

seeds.txt生成

```bash
python3 ./tools/seeds.py > ./tools/seeds.txt
```

# AHC017

市民の不満を最小化する工事計画問題．
