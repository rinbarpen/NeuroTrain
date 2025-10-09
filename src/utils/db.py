from pathlib import Path
import sqlalchemy
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Table
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import pandas as pd
import fastparquet

class DB:
    def __init__(self, db_name: str, *, verbose=False):
        self.db_name = db_name
        self.engine = sqlalchemy.create_engine(f'sqlite:///{db_name}.db', echo=verbose)
        self.Base = sqlalchemy.orm.declarative_base()
        self.Base.metadata.create_all(self.engine)
        self.verbose = verbose
    
    def __enter__(self):
        self.session = sqlalchemy.orm.sessionmaker(bind=self.engine)()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.commit()
        self.session.close()

    def _normalize_to_df_and_payload(self, data: list | pd.DataFrame):
        """
        将输入数据标准化为 (DataFrame, list[dict]) 形式，便于下游统一处理。
        - data 支持 DataFrame 或 list[dict]
        """
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(data or [])
        payload = df.to_dict(orient='records')
        return df, payload

    def _reflect_table(self, table_name: str):
        """反射加载或更新 metadata 中的表定义。"""
        return sqlalchemy.Table(
            table_name,
            self.Base.metadata,
            autoload_with=self.engine,
            extend_existing=True,
        )

    def _is_no_such_table_error(self, e: Exception) -> bool:
        """判断是否为“表不存在”的错误（兼容反射期和执行期）。"""
        if isinstance(e, sqlalchemy.exc.NoSuchTableError):
            return True
        if isinstance(e, sqlalchemy.exc.OperationalError) and 'no such table' in str(e).lower():
            return True
        return False

    def _ensure_table_schema(self, table_name: str, df_schema: pd.DataFrame) -> bool:
        """
        基于 df_schema 创建表结构，不写入数据。返回 True 表示已创建/更新结构，False 表示无法创建（例如 df 为空）。
        说明：使用 head(0).to_sql 仅创建表结构，避免重复写入；随后反射到 metadata。
        """
        if df_schema is None or df_schema.empty:
            return False
        df_schema.head(0).to_sql(table_name, self.engine, if_exists='replace', index=False)
        # 反射更新 metadata
        self._reflect_table(table_name)
        return True

    def _delete_all_rows(self, table_name: str):
        """删除整个表的所有数据（若表不存在，由调用方捕获异常后忽略）。"""
        table = self.Base.metadata.tables.get(table_name) or self._reflect_table(table_name)
        self.session.execute(table.delete())

    def insert(self, table_name: str, data: list | pd.DataFrame):
        """
        高效插入：
        - DataFrame 走 pandas.to_sql(if_exists='append', method='multi')，批量写入性能更好；
        - list[dict] 走 SQLAlchemy 批量插入；
        - 若遇到“表不存在”，先创建表结构再重试（仅创建 schema，不写入数据）。
        """
        df, payload = self._normalize_to_df_and_payload(data)
        if not payload:
            return

        # DataFrame 走最快路径：pandas 批量写入
        if isinstance(data, pd.DataFrame):
            try:
                df.to_sql(table_name, self.engine, if_exists='append', index=False, method='multi')
            except (sqlalchemy.exc.NoSuchTableError, sqlalchemy.exc.OperationalError) as e:
                if self._is_no_such_table_error(e):
                    # 创建空表结构后重试批量追加
                    if not self._ensure_table_schema(table_name, df):
                        return
                    df.to_sql(table_name, self.engine, if_exists='append', index=False, method='multi')
                else:
                    raise
            return

        # list[dict] 使用 ORM 批量插入
        try:
            table = self.Base.metadata.tables.get(table_name) or self._reflect_table(table_name)
            self.session.execute(table.insert(), payload)
            self.session.commit()
        except (sqlalchemy.exc.NoSuchTableError, sqlalchemy.exc.OperationalError) as e:
            if self._is_no_such_table_error(e):
                self.session.rollback()
                if not self._ensure_table_schema(table_name, df):
                    return
                table = self.Base.metadata.tables[table_name]
                self.session.execute(table.insert(), payload)
                self.session.commit()
            else:
                self.session.rollback()
                raise
        except Exception:
            self.session.rollback()
            raise

    def remove_table(self, table_name: str):
        """
        清空表数据：遵循“先执行 SQL，再按需处理异常”的策略，避免多余预检查。
        表不存在时忽略（视为已空）。
        """
        try:
            table = self.Base.metadata.tables.get(table_name) or self._reflect_table(table_name)
            self.session.execute(table.delete())
            self.session.commit()
        except (sqlalchemy.exc.NoSuchTableError, sqlalchemy.exc.OperationalError) as e:
            if self._is_no_such_table_error(e):
                self.session.rollback()
                return
            self.session.rollback()
            raise
        except Exception:
            self.session.rollback()
            raise

    def update(self, table_name: str, data: list | pd.DataFrame):
        """
        全量更新（替换）表内容：
        - 语义明确、实现高效：用 pandas.to_sql(if_exists='replace') 一步完成结构推断与数据写入；
        - 若表不存在会自动创建；适合“以新数据覆盖旧数据”的场景；
        - 如需按主键/条件增量更新，请另行设计 upsert 逻辑。
        """
        df, payload = self._normalize_to_df_and_payload(data)
        if not payload:
            return
        try:
            df.to_sql(table_name, self.engine, if_exists='replace', index=False, method='multi')
            # 更新 metadata，保持后续操作一致
            self._reflect_table(table_name)
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise

    def create_table_from_df(self, table_name: str, df: pd.DataFrame):
        df.to_sql(table_name, self.engine, if_exists='replace', index=False)
        self.session.flush()
        self.session.commit()
    
    def query(self, table_name: str, columns: list[str] = None):
        table = self.Base.metadata.tables[table_name]
        if columns is None:
            columns = table.columns
        return self.session.query(*columns).all()
    
    def table(self, table_name: str):
        return DBTable(table_name, self)

class DBTable:
    def __init__(self, table_name: str, db: 'DB'):
        self.table_name = table_name
        self.db = db
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.session.commit()
        self.db.session.close()

    def query(self, columns: list[str] = None):
        return self.db.query(self.table_name, columns)

    def insert(self, data: list[dict] | pd.DataFrame):
        self.db.insert(self.table_name, data)

    def update(self, data: list[dict] | pd.DataFrame):
        self.db.update(self.table_name, data)
    
    def remove(self):
        self.db.remove_table(self.table_name)
