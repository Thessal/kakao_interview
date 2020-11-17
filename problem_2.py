from sqlalchemy import *
import pandas as pd

verbose = False


def _print(*msg):
    print('*', *list(msg))


class Problem2:

    def _load_data(self, path, table_name):
        """
        read csv file to sql

        :param path: csv file path (with trailing slash)
        """
        _print(f'Initializing DB {table_name}')
        df = pd.read_csv(f'{path}{table_name}.csv')
        df.sort_values(df.columns[0], ascending=True, inplace=True)
        df.to_sql(table_name, index=False, con=self.engine, if_exists='append')
        df_test = pd.read_sql(f'SELECT * FROM {table_name}', self.engine)
        if not df.set_index(df.columns[0]).equals(df_test.set_index(df_test.columns[0])):
            print(f'{table_name} : Error')
            print(len(df), len(df_test))
            print(df.compare(df_test))

    def _load_schema(self):
        """
        define db schema

        """
        _print("Initializing Schema")

        Table('users', self.metadata,
              Column('user_id', String(15), primary_key=True),
              Column('gender_cd', Integer, nullable=False),
              Column('age', Integer, nullable=False),
              Column('foreigner_yn', String(1), nullable=False),
              Column('os_type', String(1), nullable=False)
              )

        Table('dutchpay_claim', self.metadata,
              Column('claim_id', Integer, primary_key=True),
              Column('claim_at', DateTime, nullable=False),
              Column('claim_user_id', String(15), ForeignKey("users.user_id"), nullable=False)
              )

        Table('dutchpay_claim_detail', self.metadata,
              Column('claim_detail_id', Integer, primary_key=True),
              Column('claim_id', Integer, ForeignKey("dutchpay_claim.claim_id"),
                     nullable=False),
              Column('recv_user_id', String(15), ForeignKey("users.user_id"), nullable=False),
              Column('claim_amount', Integer, nullable=False),
              Column('send_amount', Integer, nullable=True),
              Column('status', Enum(*['CLAIM', 'CHECK', 'SEND']), nullable=False)
              )

        Table('a_payment_trx', self.metadata,
              Column('id', String(37), primary_key=True),
              Column('transaction_id', Integer, nullable=False),
              Column('transacted_at', DateTime, nullable=False),
              Column('payment_action_type', Enum(*['PAYMENT', 'CANCEL']), nullable=False),
              Column('user_id', String(15), ForeignKey("users.user_id"), nullable=False),
              Column('amount', Integer, nullable=False)
              )

    def _calc_validate(self):
        """
        calculate validation dataframe

        :returns: pandas dataframe
        """
        # For validation

        df_target = pd.read_sql_query('select * from dutchpay_claim', con=self.engine)
        df_target = df_target.claim_user_id
        df_dutch_uids = df_target.copy()

        df_target = pd.read_sql_query('select * from a_payment_trx', con=self.engine)
        # df_target = df_target[ df_target.payment_action_type == 'PAYMENT' ]
        df_target = df_target[df_target.user_id.isin(df_dutch_uids.values)]

        df_target = df_target[df_target.transacted_at >= '2019-12-01 00:00:00']
        df_target = df_target[
            ((df_target.transacted_at < '2020-03-01 00:00:00') & (df_target.payment_action_type == 'CANCEL'))
            | ((df_target.transacted_at < '2020-01-01 00:00:00') & (df_target.payment_action_type == 'PAYMENT'))
            ]
        df_target = df_target.pivot(columns='payment_action_type', index='transaction_id', values=['user_id', 'amount'])
        df_target.dropna(subset=[('amount', 'PAYMENT')], inplace=True)
        df_target['amount', 'CANCEL'].fillna(0, inplace=True)
        df_target = pd.DataFrame({
            'user_id': df_target.user_id.PAYMENT.fillna(df_target.user_id.CANCEL),
            'net_amount': df_target.amount.PAYMENT - df_target.amount.CANCEL
        })
        df_target = df_target[df_target.net_amount > 0]
        df_target = df_target.groupby('user_id').sum()
        df_target = df_target[df_target.values >= 10000]

        self.df_validate = df_target.drop(columns=['net_amount']).copy()

    def __init__(self, path='data/DS_사전과제_v2/'):
        self.engine = create_engine('sqlite://', echo=False)
        self.metadata = MetaData()
        self._load_schema()
        self.metadata.create_all(self.engine)

        for table_name in self.engine.table_names():
            self._load_data(path, table_name)

        self._calc_validate()

    def solve(self, sql_file='problem_2.sql'):

        _print("Making SQL Query")
        with open(sql_file, "r") as fp:
            sql_text = fp.read()
        print(sql_text)
        df = pd.read_sql_query(sql_text, con=self.engine)

        _print("Result")
        print(df)

        _print("Validate")
        print(df.set_index(['user_id']).equals(self.df_validate))
