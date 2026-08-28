# 2 extract logic version and cell color
import requests
import pandas as pd
from bs4 import BeautifulSoup
import io
import re
import time

def normalize_inline_spaces(text):
    """
    规范行内空白：
    - 合并空格/制表符，但保留换行符\n
    - 移除英文标点前空格
    - 规范括号与破折号附近空格
    - 将不间断空格替换为普通空格
    """
    if not isinstance(text, str):
        return text
    # 将 NBSP 统一为普通空格
    text = text.replace('\xa0', ' ')
    # 合并空格与制表符（保留换行）
    text = re.sub(r'[ \t]+', ' ', text)
    # 去掉标点前空格
    text = re.sub(r'[ \t]+([,;:.!?%])', r'\1', text)
    # 括号内外空格规范
    text = re.sub(r'([\(\[\{])[ \t]+', r'\1', text)
    text = re.sub(r'[ \t]+([\)\]\}])', r'\1', text)
    # en dash 两侧空格规范
    text = re.sub(r'[ \t]*–[ \t]*', ' – ', text)
    return text.strip()

def extract_color_from_style(style_str):
    """
    从style属性字符串中提取背景颜色，并忽略CSS变量。
    """
    if not style_str:
        return None
    # 使用正则表达式匹配 background-color 或 background 属性
    match = re.search(r'background(?:-color)?\s*:\s*([^;]+)', style_str)
    if match:
        color = match.group(1).strip()
        # 检查提取的是否是CSS变量 (var(...))，如果是则忽略
        if color.startswith('var('):
            return None
        return color
    return None


def get_cleaned_cell_text(cell):
    """辅助函数：清理单元格，移除坐标并提取文本。"""
    # 一次性移除所有不需要的内联元素，包括内嵌表格、坐标、折叠按钮等
    for unwanted_element in cell.select('table, .geo-nondefault, .geo-multi-punct, .mw-collapsible-toggle'):
        unwanted_element.decompose()
    
    # 移除隐藏元素（避免抓取 display:none / visibility:hidden 中的 ISO 日期等）
    for hidden in cell.find_all(True, style=lambda s: isinstance(s, str) and re.search(r'(display\s*:\s*none|visibility\s*:\s*hidden)', s, flags=re.I)):
        hidden.decompose()

    # 处理换行
    for br in cell.find_all('br'):
        br.replace_with('\n')
    
    # 处理列表
    list_items = cell.select('ul > li')
    if list_items:
        lines = [normalize_inline_spaces(li.get_text(" ", strip=True)) for li in list_items]
        return '\n'.join(lines)
    else:
        # 使用分隔符确保链接与前后文本之间的空格得以保留
        text = cell.get_text(" ", strip=True)
        return normalize_inline_spaces(text)


def _sanitize_table_span_attributes(table):
    """
    规范化表格中单元格的 rowspan/colspan 属性值，移除诸如 "2;" 这类带非数字字符的情况，
    以避免 pandas.read_html 在将其转换为 int 时抛出异常。

    规则：
    - 提取属性值中的第一个连续数字串；
    - 若未提取到数字，则将该属性设置为 '1'；
    - 若提取到为 '0'，也回退为 '1'（避免无效跨度）。
    """
    for cell in table.find_all(['td', 'th']):
        for attr in ('rowspan', 'colspan'):
            raw_val = cell.get(attr)
            if raw_val is None:
                continue
            val_str = str(raw_val)
            match = re.search(r'\d+', val_str)
            if match:
                clean_val = match.group(0) or '1'
                if clean_val == '0':
                    clean_val = '1'
                cell[attr] = clean_val
            else:
                # 不存在数字，使用 1 作为安全默认值
                cell[attr] = '1'


def get_wikipedia_tables(url, with_color=True, retries=3, backoff_factor=1):
    """
    从给定的维基百科URL中提取所有表格，并附带其上下文信息。增加了重试机制。

    Args:
        url (str): 维基百科页面的URL。
        with_color (bool): 是否提取单元格颜色。
        retries (int): 失败时的重试次数。
        backoff_factor (float): 指数回退的基准秒数。

    Returns:
        tuple: 一个元组，包含处理状态和结果。
              - ("SUCCESS", list): 成功，返回一个字典的列表。每个字典代表一个表格。
              - ("ERROR", str): 失败，返回错误信息。
    """
    headers = {
        'Connection': 'close',
        'User-Agent': 'CoG-Bot/1.0 (Academic Research; +https://github.com/anonymous/CoG)'
    }
    
    response = None
    last_exception = None
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=180)
            response.raise_for_status()
            break  # Success
        except requests.exceptions.RequestException as e:
            last_exception = e
            print(f"Table Retrieval Error on attempt {attempt + 1}/{retries} for {url}: {e}")
            if attempt < retries - 1:
                sleep_time = backoff_factor * (2 ** attempt)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)

    if response is None:
        error_message = f"Failed to retrieve tables from {url} after {retries} attempts. Last error: {last_exception}"
        return ("ERROR", error_message)

    try:
        soup = BeautifulSoup(response.text, 'lxml')
        all_tables = soup.find_all('table', class_=['wikitable', 'infobox'])
        
        results_list = []

        for table in all_tables:
            # 在处理任何表格之前，首先移除所有的引用上标
            for sup in table.find_all('sup', class_='reference'):
                sup.decompose()

            table_info = {}
            is_infobox = 'infobox' in (table.attrs or {}).get('class', [])

            if is_infobox:
                # --- 针对 Infobox 的专用解析逻辑 ---
                table_info['top_section'] = "Summary"
                caption = table.find('caption')
                caption_text = caption.get_text(strip=True) if caption else None
                table_info['table_name'] = caption_text if caption_text else "Summary Infobox"

                # 预先移除所有不需要的行
                for row in table.find_all('tr'):
                    if row.select_one('.infobox-image, .infobox-full-data, .infobox-above'):
                        row.decompose()

                labels = []
                values = []
                
                for row in table.find_all('tr'):

                    header_cell = row.find('th')
                    data_cells = row.find_all('td')

                    if header_cell:
                        label = get_cleaned_cell_text(header_cell)
                        if not data_cells:  # 只有 th 的行，通常是子标题
                            labels.append(label)
                            values.append(label)
                        else:  # 有 th 和一个或多个 td 的行
                            # 第一个 td 和 th 配对
                            labels.append(label)
                            values.append(get_cleaned_cell_text(data_cells[0]))
                            # 后续的 td 作为延续行
                            for cell in data_cells[1:]:
                                labels.append("")
                                values.append(get_cleaned_cell_text(cell))
                    elif data_cells:  # 没有 th 但有 td 的行
                        # 每个 td 都是一个独立的延续行
                        for cell in data_cells:
                            labels.append("")
                            values.append(get_cleaned_cell_text(cell))
                
                if labels and values:
                    # 使用固定的、清晰的列名创建DataFrame
                    df = pd.DataFrame({"Attribute": labels, "Value": values})
                    # 清理完全为空或只有空白的行
                    df = df[df['Attribute'].astype(str).str.strip().astype(bool) | df['Value'].astype(str).str.strip().astype(bool)]
                    table_info['dataframe'] = df
                    results_list.append(table_info)
            
            else: # --- 针对 Wikitable 的现有逻辑 ---
                # 先清洗 rowspan/colspan，避免 pandas.read_html 在解析时因为诸如 "6;" 报错
                _sanitize_table_span_attributes(table)
                # 1. 查找顶层章节 (Top Section - h2)
                table_info['top_section'] = "Unknown" # 默认值
                top_section_header = table.find_previous('h2')
                if top_section_header:
                    table_info['top_section'] = top_section_header.get_text(strip=True).replace('[edit]', '').strip()
                
                # 2. 查找标题作为名称，优先使用<caption>，其次是最近的标题
                caption = table.find('caption')
                caption_text = caption.get_text(strip=True) if caption else None
                if caption_text:
                    table_info['table_name'] = caption_text
                else:
                    nearest_header = table.find_previous(['h2', 'h3', 'h4', 'h5', 'h6'])
                    if nearest_header:
                        table_info['table_name'] = nearest_header.get_text(strip=True).replace('[edit]', '').strip()
                    else:
                        table_info['table_name'] = "Untitled Table" # 如果都没找到，使用默认值

                # 在解析为DataFrame之前，处理单元格颜色
                if with_color:
                    for cell in table.find_all(['td', 'th']):
                        style = cell.get('style')
                        if style:
                            color = extract_color_from_style(style)
                            if color:
                                # 将颜色信息作为文本追加到单元格内容中
                                cell.append(f" [cell-color: {color}]")

                try:
                    # 步骤 1: 预处理 HTML。将 <br> 标签替换为一个独特的占位符。
                    html_string = str(table).replace('<br/>', ' |||| ').replace('<br>', ' |||| ')
                    # 步骤 2: 将修改后的 HTML 解析为 DataFrame。
                    df = pd.read_html(io.StringIO(html_string))[0]
                    
                    # 步骤 3: 如果存在多级列名 (MultiIndex)，则将其扁平化 (通用版本)。
                    if isinstance(df.columns, pd.MultiIndex):
                        new_columns = []
                        for col in df.columns:
                            # 使用列表来保持顺序，同时确保唯一性
                            unique_levels = []
                            for level in col:
                                level_str = str(level).strip()
                                # 过滤掉 'Unnamed' 并避免添加重复的级别
                                if 'Unnamed' not in level_str and level_str not in unique_levels:
                                    unique_levels.append(level_str)
                            # 将唯一的、有序的级别连接起来形成新的列名
                            new_columns.append(' '.join(unique_levels))
                        df.columns = new_columns

                    # 步骤 4: 后处理列名，将列名中的占位符替换为换行符。
                    df.columns = [
                        col.replace(' |||| ', '\n').strip() if isinstance(col, str) else col 
                        for col in df.columns
                    ]
                    
                    # 步骤 5: 后处理 DataFrame 数据。将占位符替换回换行符并清理空格。
                    df = df.replace(r'\s*\|\|\|\|\s*', '\n', regex=True)
                    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

                    table_info['dataframe'] = df
                    results_list.append(table_info)
                except Exception as e:
                    print(f"Table Parsing Error! Cannot parse '{table_info.get('table_name', 'Unknown')}' table: {e}")
                    
        # --- 后处理：为重名表格添加序号 ---
        name_counts = {}
        for info in results_list:
            name = info['table_name']
            name_counts[name] = name_counts.get(name, 0) + 1

        rename_counters = {}
        for info in results_list:
            name = info['table_name']
            if name_counts[name] > 1:
                current_count = rename_counters.get(name, 1)
                info['table_name'] = f"{name} {current_count}"
                rename_counters[name] = current_count + 1

        # --- 计算表格序列化后的大小 ---
        for info in results_list:
            df = info.get('dataframe')
            if df is not None:
                # Markdown 格式
                info['markdown_total_len'] = len(df.to_markdown(index=False))
                info['markdown_head_len'] = len(df.head(5).to_markdown(index=False))
                # CSV 格式
                info['csv_total_len'] = len(df.to_csv(index=False))
                info['csv_head_len'] = len(df.head(5).to_csv(index=False))
            else:
                # 如果没有 DataFrame，则长度为0
                info['markdown_total_len'] = 0
                info['markdown_head_len'] = 0
                info['csv_total_len'] = 0
                info['csv_head_len'] = 0
                
        return ("SUCCESS", results_list)
    except Exception as e:
        error_message = f"Table Parsing Error! Cannot parse tables for {url}: {e}"
        print(error_message)
        return ("ERROR", error_message)