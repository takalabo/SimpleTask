# SimpleTask
現在，機械学習分野の１つである深層強化学習はさまざなな分野で活用されています．ですが，深層強化学習のプログラムは気軽に動かすことができません．深層強化学習は計算リソースを大量に使い，1回の学習時間がとても長いという問題があります．この問題は深層強化学習分野への参入のハードルにもなっています．私たちが提供する SimpleTask はこれらの問題を DeNA が開発した HandyRL と組み合わせることで解決します．

* [HandyRLとは](#HandyRLとは)
* [SimpleTaskとは](#SimpleTaskとは)
* [利用方法](#利用方法)
* [実行方法](#実行方法)
* [現在実装されている機能](#現在実装されている機能)
* [ドキュメント](#ドキュメント)

## HandyRLとは
HandyRL とは，DeNA によって開発された最新の並列強化学習のフレームワークです．HandyRL では誰でも強化学習を手軽に扱え，簡単に強いAIを作ることができます．また実装が難しい並列強化学習を手軽に試すことができます．HandyRLの詳細については[こちら](https://github.com/DeNA/HandyRL)をご覧ください．

## SimpleTaskとは
SimpleTask とは，深層強化学習の基礎研究を誰でも気軽に行えることを目的として開発された深層強化学習の検証タスクです．深層強化学習の問題点を，タスク設計を工夫し，並列強化学習を手軽に扱うことができる HandyRL に組み込むことで解決します．SimpleTask の詳細については[こちら](https://drive.google.com/file/d/1cL8N-yL3mHcks0rxp2Qp1s7pynn7J6vp/view?usp=sharing)をご覧ください．

## 利用方法
まず，HandyRL を[こちら](https://github.com/DeNA/HandyRL)のページからインストールしてください．

インストールが完了しましたら，SimpleTask リポジトリをダウンロードしてください．
SimpleTask のダウンロードが完了しましたら，HandyRL のディレクトリ下に SimpleTask ディレクトリ内の一部コードを移動させます．移動させるファイルは以下のファイルです．

`SimpleTask/simpletask.py` → `HandyRL/handyrl/envs/simpletask.py`

`SimpleTask/config.yaml` → `HandyRL/config.yaml`

`simpletask.py`を `envs` ディレクトリ下に移動させ， HandyRL の`config.ymal`を SimpleTask の`config.yaml`に置き換えてください．

以上で SimpleTask を動かす準備は完了です． 実行してみましょう！

## 実行方法
基本的な実行方法は HandyRL と同じですが， `config.yaml`に SimpleTask 用のパラメータが増えています． 具体的な手順を以下に記載します．

#### Step 1: パラメータを設定する
`config.yaml`のパラメータをトレーニングに合わせて以下のように設定します．環境を simpletask ，バッチサイズを64でトレーニングを実行する場合は，以下のように設定します．

```yaml
env_args:
    env: 'simpletask'

train_args:
    ...
    batch_size: 64
    ...
```

注意: SimpleTask のパラメータは， [こちら](docs/parameters.md)を参照してください．


#### Step 2: トレーニング
パラメータを設定したら， 以下のコマンドを実行してトレーニングを開始します． トレーニングされたモデルは， `config.yaml`の`update_episodes`毎に`models`に保存されます．
```
python main.py --train
```


#### Step 3: 評価
トレーニング後，任意のモデルに対して評価できます．以下のコマンドは，エポック1のモデルを4プロセスで100ゲーム分評価します．
```
python main.py --eval models/1.pth 100 4
```
注意: デフォルトの対戦相手AIは`evaluation.py`で実装されたランダムなエージェントです．また，自分で任意のエージェントに変更することができます．SimpleTask は現段階においては対戦ゲームではないです．そのため動作が不安定なことがあるので，その点に関しましてはあらかじめご了承ください．今後のアップデートで対応いたします．

## 現在実装されている機能
* 超平面次元数の設定
* 最大深度数の設定
* 報酬の場所の設定
* 報酬の量の設定
* 複数報酬の設定（報酬の量は全て同じ）
* スタート地点の非ランダム化
* POMDP （途中報酬）の設定

## ドキュメント

* [**SimpleTask Parameters**](docs/parameters.md) `config.yaml`で SimpleTask のタスク設定が可能．パラメータの詳細はここのページを参照してください．
* [**JSAI2022発表論文**](https://drive.google.com/file/d/1cL8N-yL3mHcks0rxp2Qp1s7pynn7J6vp/view?usp=sharing) ここに SimpleTask の詳細な説明があります．
* [**JSAI2022発表スライド**](https://docs.google.com/presentation/d/19ZVz0u7HMlmrcXrbripUFiIgjLlLBqmba2TWHw1-hfg/edit?usp=sharing) 発表論文を要約した内容が書いてあります．
